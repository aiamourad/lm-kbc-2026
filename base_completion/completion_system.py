#!/usr/bin/env python3
# The completion formats, the sampling, and the parsing, for all six relations.
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm import tqdm

from client import Client, ServerUnavailable
from aggregation import cluster_vote, format_number, parse_number
from aggregation import SET_SPECS, aggregate

NUMERIC_SENT = {
    "hasArea": "The total area of {s} is {v} square kilometres.",
    "hasCapacity": "The seating capacity of {s} is {v}.",
}

SET_TEMPLATES = {
    "countryLandBordersCountry":
        "Countries sharing a land border with {s}: {v}",
    "companyTradesAtStockExchange":
        "{s} stock exchange listings: {v}",
    "personHasCityOfDeath":
        "City where {s} died (or 'none' if alive as of July 2026): {v}",
    "awardWonBy":
        "Recipients of {s}: {v}",
}

EMPTY_WORD = "none"
SET_SHOTS = {"awardWonBy": 4}
SET_MAXTOK = {"awardWonBy": 400}
DEFAULT_SET_SHOTS = 8
DEFAULT_SET_MAXTOK = 96


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def gold_labels(row: dict) -> list[str]:
    return [g[0] if isinstance(g, list) else g for g in row["ObjectEntities"]]


def numeric_prefix(relation: str, train_rows: list[dict]) -> str:
    sent = NUMERIC_SENT[relation]
    pool = []
    for r in train_rows:
        if r["Relation"] != relation or not r["ObjectEntities"]:
            continue
        try:
            pool.append((r["SubjectEntity"],
                         float(str(gold_labels(r)[0]).replace(",", ""))))
        except ValueError:
            continue
    pool.sort(key=lambda x: x[1])
    step = max(1, len(pool) // 5)
    picked = pool[::step][:5]
    random.Random(7).shuffle(picked)
    lines = [sent.format(s=s_, v=(int(v) if v == int(v) else v))
             for s_, v in picked]
    return "\n".join(lines) + "\n" + sent.split(" is ")[0].replace("{s}", "{subject}") + " is"


def set_prefix(relation: str, train_rows: list[dict], shots: int) -> str:
    template = SET_TEMPLATES[relation]
    pool = [r for r in train_rows if r["Relation"] == relation]
    empties = [r for r in pool if not r["ObjectEntities"]]
    nonempties = [r for r in pool if r["ObjectEntities"]]
    rng = random.Random(11)
    n_empty = min(len(empties), max(1, shots // 3)) if empties else 0
    chosen = rng.sample(empties, n_empty) + rng.sample(
        nonempties, min(shots - n_empty, len(nonempties)))
    rng.shuffle(chosen)
    lines = []
    for r in chosen:
        labels = gold_labels(r)
        value = ", ".join(labels[:12]) if labels else EMPTY_WORD
        lines.append(template.format(s=r["SubjectEntity"], v=value))
    return "\n".join(lines) + "\n" + template.format(s="{subject}", v="").rstrip()


def parse_set_line(text: str) -> list[str]:
    line = str(text or "").strip().split("\n")[0].strip()
    if not line or line.lower().startswith(EMPTY_WORD):
        return []
    parts = [p.strip(" .;") for p in line.split(",")]
    return [p for p in parts if p and len(p) < 80]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--set-shots", type=int, default=None,
                        help="Override exemplar count for set relations.")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--award-support", type=float, default=None)
    parser.add_argument("--company-support", type=float, default=None)
    parser.add_argument("--border-support", type=float, default=None)
    parser.add_argument("--death-support", type=float, default=None)
    parser.add_argument("--relations", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dump-samples", default=None)
    args = parser.parse_args()

    support = {
        "awardWonBy": args.award_support,
        "companyTradesAtStockExchange": args.company_support,
        "countryLandBordersCountry": args.border_support,
        "personHasCityOfDeath": args.death_support,
    }

    train_rows = read_jsonl(args.train)
    rows = read_jsonl(args.input)
    if args.relations:
        wanted = {r.strip() for r in args.relations.split(",") if r.strip()}
        rows = [r for r in rows if r["Relation"] in wanted]
    if args.limit:
        rows = rows[: args.limit]

    prefixes: dict[str, str] = {}
    for rel in NUMERIC_SENT:
        prefixes[rel] = numeric_prefix(rel, train_rows)
    for rel in SET_TEMPLATES:
        shots = args.set_shots or SET_SHOTS.get(rel, DEFAULT_SET_SHOTS)
        prefixes[rel] = set_prefix(rel, train_rows, shots)

    checkpoint = Path(str(args.output) + ".checkpoint.jsonl")
    done: dict[tuple[str, str], dict] = {}
    resume = not args.no_resume and not args.dump_samples
    if resume and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                done[(rec["SubjectEntity"], rec["Relation"])] = rec
            except (json.JSONDecodeError, KeyError):
                continue

    pending = [i for i, r in enumerate(rows)
               if (r["SubjectEntity"], r["Relation"]) not in done]
    print(f"{len(rows)} rows, {len(rows)-len(pending)} done, {len(pending)} to run",
          file=sys.stderr)

    client = Client(base_url=args.base_url, model=args.model)
    client.wait_until_ready()

    def predict(row: dict) -> tuple[dict, dict | None]:
        relation, subject = row["Relation"], row["SubjectEntity"]
        prefix = prefixes[relation].format(subject=subject)
        numeric = relation in NUMERIC_SENT
        maxtok = 32 if numeric else SET_MAXTOK.get(relation, DEFAULT_SET_MAXTOK)

        samples = []
        for i in range(args.samples):
            try:
                text = client.complete(
                    prompt=prefix, temperature=args.temperature,
                    max_tokens=maxtok, seed=1300 + i, stop=["\n"])
            except ServerUnavailable:
                raise
            except Exception:
                continue
            if numeric:
                v = parse_number(text, detect_units=True)
                if v is not None and v > 0:
                    samples.append(v)
            else:
                samples.append(parse_set_line(text))

        if numeric:
            value, _ = cluster_vote(samples)
            objects = [format_number(value)] if value is not None else []
            trace = ({"SubjectEntity": subject, "Relation": relation,
                      "samples": samples} if args.dump_samples else None)
        else:
            spec = SET_SPECS[relation]
            objects = aggregate(samples, spec,
                                min_support=support.get(relation))
            trace = ({"SubjectEntity": subject, "Relation": relation,
                      "samples": samples} if args.dump_samples else None)

        return ({"SubjectEntity": subject, "Relation": relation,
                 "ObjectEntities": objects}, trace)

    lock = Lock()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    handle = checkpoint.open("a" if (resume and done) else "w", encoding="utf-8")
    dump = Path(args.dump_samples).open("w", encoding="utf-8") if args.dump_samples else None

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(predict, rows[i]): i for i in pending}
            for f in tqdm(as_completed(futs), total=len(futs), desc="rows"):
                i = futs[f]
                row = rows[i]
                trace = None
                try:
                    rec, trace = f.result()
                except ServerUnavailable as e:
                    print(f"\nFATAL: {e}\ncheckpoint kept at {checkpoint}",
                          file=sys.stderr)
                    for pf in futs:
                        pf.cancel()
                    raise SystemExit(2)
                except Exception as e:
                    print(f"\nrow {i} failed: {e}", file=sys.stderr)
                    rec = {"SubjectEntity": row["SubjectEntity"],
                           "Relation": row["Relation"], "ObjectEntities": []}
                done[(rec["SubjectEntity"], rec["Relation"])] = rec
                with lock:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    handle.flush()
                    if dump is not None and trace is not None:
                        dump.write(json.dumps(trace, ensure_ascii=False) + "\n")
                        dump.flush()

        out = [done.get((r["SubjectEntity"], r["Relation"]))
               or {"SubjectEntity": r["SubjectEntity"],
                   "Relation": r["Relation"], "ObjectEntities": []}
               for r in rows]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n",
            encoding="utf-8")
        print(f"wrote {len(out)} predictions to {args.output}", file=sys.stderr)
    finally:
        handle.close()
        if dump is not None:
            dump.close()
        client.close()


if __name__ == "__main__":
    main()
