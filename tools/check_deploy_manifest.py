"""
Verify the Dockerfile actually copies every local module the app imports.

This exists because it has already gone wrong: screener_v2/ was added and
imported by main.py, but the Dockerfile's COPY list still named only
main.py, db.py, auth.py and web/. Everything passed locally — tests,
import checks, static analysis — because locally the package was simply
sitting in the working directory. In the container it wasn't there at all,
so uvicorn died on import and the service exited before serving a request.

Nothing in the test suite can catch that, because the bug isn't in the
code; it's in the gap between the repository and the image built from it.

Run: python3 tools/check_deploy_manifest.py [Dockerfile]
Exit code is non-zero if a locally-imported module isn't copied.
"""
from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRYPOINTS = ("main.py",)


def local_module_names() -> set[str]:
    """Top-level importable names that live in this repo."""
    names = set()
    for item in os.listdir(ROOT):
        path = os.path.join(ROOT, item)
        if item.endswith(".py"):
            names.add(item[:-3])
        elif os.path.isdir(path) and os.path.exists(os.path.join(path, "__init__.py")):
            names.add(item)
    return names


def imported_top_level(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Relative imports resolve inside an already-copied package.
            if node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
    return imported


def copied_sources(dockerfile: str) -> set[str]:
    """Source paths named on COPY lines, normalized to a bare name."""
    sources = set()
    with open(dockerfile, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line.upper().startswith("COPY "):
                continue
            tokens = [t for t in line.split()[1:] if not t.startswith("--")]
            if len(tokens) < 2:
                continue
            for token in tokens[:-1]:  # last token is the destination
                name = token.lstrip("./").rstrip("/")
                sources.add(name)
                if name.endswith(".py"):
                    sources.add(name[:-3])
    return sources


def main() -> int:
    dockerfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "Dockerfile")
    if not os.path.exists(dockerfile):
        print(f"No Dockerfile at {dockerfile}")
        return 1

    local = local_module_names()
    copied = copied_sources(dockerfile)
    missing = []

    for entrypoint in ENTRYPOINTS:
        entry_path = os.path.join(ROOT, entrypoint)
        if not os.path.exists(entry_path):
            continue
        for name in sorted(imported_top_level(entry_path) & local):
            if name not in copied:
                missing.append(f"{entrypoint} imports {name!r}, which no COPY line in the Dockerfile includes")

    # The entrypoints themselves have to be copied too.
    for entrypoint in ENTRYPOINTS:
        if os.path.exists(os.path.join(ROOT, entrypoint)) and entrypoint not in copied:
            missing.append(f"{entrypoint} is the entrypoint but is not copied into the image")

    if missing:
        print("DEPLOY MANIFEST VIOLATIONS:")
        for item in missing:
            print(f"  - {item}")
        print("\nThe image would start, fail on import, and exit before serving.")
        return 1

    print(f"ok    every local module imported by {', '.join(ENTRYPOINTS)} is copied into the image")
    return 0


if __name__ == "__main__":
    sys.exit(main())
