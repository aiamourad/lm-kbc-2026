#!/usr/bin/env python3
# Validates a prediction file and zips it for submission.
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import zipfile
from pathlib import Path

NUMERIC_RELATIONS = {"hasArea", "hasCapacity"}

APOSTROPHE_LIKE = set("'’‘ʻʼʹ`´")
ASCII_SYMBOLS = set("+$<=>|~^")


def normalize(text: str) -> str:
    text = "".join(c for c in str(text).strip() if c not in APOSTROPHE_LIKE)
    text = unicodedata.normalize("NFKD", text).casefold()
    out = []
    for c in text:
        if c in APOSTROPHE_LIKE or unicodedata.combining(c):
            continue
        out.append(
            " " if c in ASCII_SYMBOLS or unicodedata.category(c).startswith("P") else c
        )
    return " ".join("".join(out).split())


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{number}: invalid JSON: {error}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions", required=True)
    parser.add_argument("-g", "--gold", required=True, help="test.jsonl")
    parser.add_argument("-o", "--out", required=True, help="Output .zip")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Emit empty predictions for uncovered rows instead of failing.",
    )
    parser.add_argument(
        "--allow-all-empty",
        default="",
        help="Comma-separated relations that are legitimately empty on every "
             "row (e.g. a fitted always-empty policy). Naming them explicitly "
             "keeps the check live for every other relation, so a crash "
             "elsewhere is still caught.",
    )
    args = parser.parse_args()

    predictions = read_jsonl(Path(args.predictions))
    gold = read_jsonl(Path(args.gold))

    problems: list[str] = []

    by_key: dict[tuple[str, str], dict] = {}
    for row in predictions:
        for field in ("SubjectEntity", "Relation", "ObjectEntities"):
            if field not in row:
                problems.append(f"row missing {field}: {row}")
                break
        else:
            key = (row["SubjectEntity"], row["Relation"])
            if key in by_key:
                problems.append(f"duplicate row for {key}")
            by_key[key] = row

    gold_keys = [(r["SubjectEntity"], r["Relation"]) for r in gold]
    missing = [key for key in gold_keys if key not in by_key]

    if missing and not args.allow_missing:
        problems.append(
            f"{len(missing)} test rows have no prediction "
            f"(first: {missing[0]}); pass --allow-missing to emit them empty"
        )

    extra = [key for key in by_key if key not in set(gold_keys)]
    if extra:
        problems.append(f"{len(extra)} predicted rows are not in the gold file "
                        f"(first: {extra[0]})")

    records: list[dict] = []
    stats: dict[str, dict[str, int]] = {}

    for key in gold_keys:
        row = by_key.get(key)
        objects = list(row["ObjectEntities"]) if row else []

        flat: list[str] = []
        seen: set[str] = set()

        for obj in objects:
            if isinstance(obj, list):
                if not obj:
                    continue
                obj = obj[0]
            if not isinstance(obj, str):
                obj = str(obj)
            obj = obj.strip()
            token = normalize(obj)
            if not obj or not token or token in seen:
                continue
            seen.add(token)
            flat.append(obj)

        relation = key[1]
        bucket = stats.setdefault(relation, {"rows": 0, "objects": 0, "empty": 0})
        bucket["rows"] += 1
        bucket["objects"] += len(flat)
        bucket["empty"] += not flat

        if relation in NUMERIC_RELATIONS:
            for obj in flat:
                try:
                    float(obj.replace(",", ""))
                except ValueError:
                    problems.append(
                        f"{key}: numeric relation has unparseable value {obj!r}"
                    )

        records.append(
            {
                "SubjectEntity": key[0],
                "Relation": relation,
                "ObjectEntities": flat,
            }
        )

    print(f"{'relation':<32} {'rows':>5} {'objs/row':>9} {'empty':>6}")
    print("-" * 55)
    for relation in sorted(stats):
        bucket = stats[relation]
        print(
            f"{relation:<32} {bucket['rows']:>5} "
            f"{bucket['objects'] / bucket['rows']:>9.2f} {bucket['empty']:>6}"
        )
    print("-" * 55)
    total_rows = sum(b["rows"] for b in stats.values())
    total_empty = sum(b["empty"] for b in stats.values())
    print(f"{'total':<32} {total_rows:>5} {'':>9} {total_empty:>6}")

    permitted_empty = {
        name.strip() for name in args.allow_all_empty.split(",") if name.strip()
    }

    for relation in sorted(stats):
        bucket = stats[relation]
        if (
            bucket["rows"] >= 20
            and bucket["empty"] == bucket["rows"]
            and relation not in permitted_empty
        ):
            problems.append(
                f"{relation}: all {bucket['rows']} rows are empty -- suspect a "
                f"dead server mid-run. If intended, pass "
                f"--allow-all-empty {relation}"
            )

    if problems:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for problem in problems[:20]:
            print(f"  - {problem}", file=sys.stderr)
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more", file=sys.stderr)
        raise SystemExit(1)

    if missing:
        print(f"\nnote: {len(missing)} rows emitted empty (--allow-missing)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("predictions.jsonl", body)

    print(f"\nwrote {out_path} ({out_path.stat().st_size} bytes, "
          f"{len(records)} rows as predictions.jsonl)")


if __name__ == "__main__":
    main()
