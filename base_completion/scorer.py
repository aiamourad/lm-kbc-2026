# Locates and imports the organisers' unmodified evaluate.py.
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

RELPATH = Path("dataset2026-main") / "evaluate.py"


def find_official(explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("KBC_OFFICIAL_EVAL"):
        candidates.append(Path(os.environ["KBC_OFFICIAL_EVAL"]))

    here = Path(__file__).resolve()
    for directory in [here.parent, *here.parents]:
        candidates.append(directory / RELPATH)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise SystemExit(
        "could not find dataset2026-main/evaluate.py; set $KBC_OFFICIAL_EVAL"
    )


def load_official(explicit: str | None = None) -> ModuleType:
    try:
        import pandas  # noqa: F401
    except ImportError:
        sys.modules["pandas"] = ModuleType("pandas")

    path = find_official(explicit)
    spec = importlib.util.spec_from_file_location("official_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["official_evaluate"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    import argparse
    import subprocess

    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gold", required=True)
    ap.add_argument("-p", "--predictions", required=True)
    ap.add_argument("--official", default=None, help="path to evaluate.py")
    args = ap.parse_args()

    official = find_official(args.official)
    subprocess.run(
        [sys.executable, str(official), "-g", args.gold, "-p", args.predictions],
        check=True,
    )


if __name__ == "__main__":
    main()
