"""
Static pre-deploy checks for the v2 screener.

Two jobs:

1. CLASS ORDERING — every pydantic model referenced as a field type must be
   defined above the model referencing it. A NameError here is an
   import-time failure, which on this deployment means a dead worker rather
   than a handled error, so it is checked statically rather than trusted.

2. INVARIANTS — the five rules in the spec that are easy to erode one
   convenient exception at a time. These are checked against the parsed
   syntax tree, not the file text, so a docstring that merely mentions
   `perf_1d` doesn't trip them while an actual reference does.

Run: python3 tools/check_screener_v2.py
Exit code is non-zero on any violation, so it can gate a deploy.
"""
from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(ROOT, "screener_v2")

# Recent-return and reactive-signal fields. None of these may appear
# anywhere in a gate module, or anywhere inside the ranking function.
FORBIDDEN_IN_GATES = {
    "perf_1d", "perf_5d", "change_pct", "recent_return", "gainers",
    "ai_verdict", "ai_score", "ai_headline", "ai_score_band",
    "sentiment", "news_sentiment", "signal_score",
}

GATE_MODULES = ("compression.py", "strength.py", "entry.py")

# catalysts.py must not be able to reach the network at all — that is the
# structural guarantee behind "no inline fetches at scan time".
NETWORK_MODULES = {
    "yfinance", "requests", "httpx", "urllib", "urllib3", "http",
    "socket", "aiohttp", "mcp", "websockets",
}

failures: list[str] = []


def parse(path: str) -> ast.Module:
    with open(path, "r", encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=path)


def check_class_ordering() -> None:
    path = os.path.join(PACKAGE, "schemas.py")
    tree = parse(path)
    defined_at: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign) or statement.annotation is None:
                continue
            for referenced in ast.walk(statement.annotation):
                if isinstance(referenced, ast.Name) and referenced.id in defined_at:
                    continue
                if isinstance(referenced, ast.Name) and referenced.id[0].isupper():
                    # A capitalised name that isn't defined yet and isn't a
                    # typing construct is either a forward reference or a
                    # model defined further down the file.
                    if referenced.id in ("Optional", "List", "Dict", "Any", "Tuple", "Union"):
                        continue
                    if referenced.id in KNOWN_EXTERNAL:
                        continue
                    failures.append(
                        f"schemas.py: {node.name} references {referenced.id} on line "
                        f"{referenced.lineno}, which is not defined above it"
                    )
        defined_at[node.name] = node.lineno


KNOWN_EXTERNAL = {"BaseModel", "str", "int", "float", "bool"}


def _forbidden_names(node: ast.AST) -> set[str]:
    found = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in FORBIDDEN_IN_GATES:
            found.add(child.id)
        elif isinstance(child, ast.Attribute) and child.attr in FORBIDDEN_IN_GATES:
            found.add(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            continue
    return found


def check_no_recent_return_in_gates() -> None:
    for filename in GATE_MODULES:
        tree = parse(os.path.join(PACKAGE, filename))
        found = _forbidden_names(tree)
        if found:
            failures.append(
                f"{filename}: gate module references {sorted(found)} — invariant 1/4: "
                "no recent return, AI verdict or sentiment may reach a gate"
            )


def check_ranking_function() -> None:
    tree = parse(os.path.join(PACKAGE, "pipeline.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "score_candidate":
            found = _forbidden_names(node)
            if found:
                failures.append(
                    f"pipeline.py: score_candidate references {sorted(found)} — "
                    "invariant 1/4: the ranking function may not see a recent return"
                )
            return
    failures.append("pipeline.py: score_candidate not found")


def check_veto_has_no_override() -> None:
    tree = parse(os.path.join(PACKAGE, "compression.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            args = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
            suspicious = [a for a in args if any(
                word in a.lower() for word in ("override", "force", "skip", "ignore", "allow")
            )]
            if suspicious:
                failures.append(
                    f"compression.py: evaluate() takes {suspicious} — invariant 2: "
                    "the extension veto is absolute and takes no override"
                )
            return
    failures.append("compression.py: evaluate() not found")


def check_catalysts_cannot_fetch() -> None:
    tree = parse(os.path.join(PACKAGE, "catalysts.py"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    offending = imported & NETWORK_MODULES
    if offending:
        failures.append(
            f"catalysts.py: imports {sorted(offending)} — invariant 5: layer 4 reads "
            "cache only and must not be able to fetch at scan time"
        )


def check_entry_emits_no_stop_order() -> None:
    tree = parse(os.path.join(PACKAGE, "entry.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip().upper()
            if value in ("STOP", "STP", "STOP_LIMIT", "STP LMT", "OCA", "OCO", "BRACKET"):
                failures.append(
                    f"entry.py line {node.lineno}: emits a {value!r} order type — the "
                    "connector supports MARKET and LIMIT only, and an unlinked stop "
                    "alongside a limit on the same shares is a margin rejection"
                )


def check_module_sizes(limit: int = 300) -> None:
    for filename in sorted(os.listdir(PACKAGE)):
        if not filename.endswith(".py"):
            continue
        path = os.path.join(PACKAGE, filename)
        with open(path, "r", encoding="utf-8") as handle:
            lines = sum(1 for _ in handle)
        if lines > limit:
            failures.append(
                f"{filename}: {lines} lines, over the {limit}-line module limit "
                "(files are edited through GitHub's web editor on mobile, where "
                "large files truncate on paste) — split it"
            )


def main() -> int:
    checks = (
        ("class ordering", check_class_ordering),
        ("no recent return in gates", check_no_recent_return_in_gates),
        ("ranking function inputs", check_ranking_function),
        ("veto has no override", check_veto_has_no_override),
        ("catalysts cannot fetch", check_catalysts_cannot_fetch),
        ("entry emits no stop order", check_entry_emits_no_stop_order),
        ("module sizes", check_module_sizes),
    )
    for label, check in checks:
        before = len(failures)
        try:
            check()
        except Exception as e:
            failures.append(f"{label}: check itself failed: {type(e).__name__}: {e}")
        print(f"  {'FAIL' if len(failures) > before else 'ok  '}  {label}")

    if failures:
        print("\nVIOLATIONS:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll static checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
