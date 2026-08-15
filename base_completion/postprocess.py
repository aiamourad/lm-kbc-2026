# The awardWonBy rescue, the existence check, alias deduplication, and applying them.
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from client import Client, ServerUnavailable
from aggregation import normalize

QUESTION = {
    "awardWonBy": "Did {c} receive the {s}?",
    "personHasCityOfDeath": "Did {s} die in {c}?",
    "companyTradesAtStockExchange": "Is {s} listed on the {c}?",
}

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())

def read(p: str) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

def alias_sets(row: dict) -> list[set[str]]:
    out = []
    for e in row["ObjectEntities"]:
        aliases = e if isinstance(e, list) else [e]
        out.append({norm(a) for a in aliases})
    return out

def _rescue_prefix(relation: str, train_rows: list[dict], shots: int) -> str:
    q = QUESTION[relation]
    pool = [r for r in train_rows if r["Relation"] == relation and r["ObjectEntities"]]
    rng = random.Random(23)
    lines = []
    for _ in range(shots // 2):
        r = rng.choice(pool)
        ents = r["ObjectEntities"]
        e = rng.choice(ents)
        label = e[0] if isinstance(e, list) else e
        lines.append(q.format(c=label, s=r["SubjectEntity"]) + " yes")
        other = rng.choice(pool)
        tries = 0
        while tries < 20:
            oe = rng.choice(other["ObjectEntities"])
            olabel = oe[0] if isinstance(oe, list) else oe
            if norm(olabel) not in {norm(a) for al in alias_sets(r) for a in al}:
                break
            other = rng.choice(pool); tries += 1
        lines.append(q.format(c=olabel, s=r["SubjectEntity"]) + " no")
    rng.shuffle(lines)
    return "\n".join(lines) + "\n"

def rescue_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="completion_system --dump-samples file")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--relation", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--min-count", type=int, default=1,
                    help="Only verify candidates appearing in >= this many samples.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gold = {(r["SubjectEntity"], r["Relation"]): r for r in read(args.gold)}
    train_rows = read(args.train)
    dumps = [r for r in read(args.dump) if r["Relation"] == args.relation]
    prefix = _rescue_prefix(args.relation, train_rows, args.shots)
    print(f"--- verification prefix ---\n{prefix}---", file=sys.stderr)

    tasks = []
    for rec in dumps:
        g = gold.get((rec["SubjectEntity"], args.relation))
        if g is None:
            continue
        m = len(rec["samples"]) or 1
        counts, surface = defaultdict(int), {}
        for s in rec["samples"]:
            for x in set(map(str, s)):
                n = norm(x)
                counts[n] += 1
                surface.setdefault(n, x)
        gold_aliases = set().union(*alias_sets(g)) if g["ObjectEntities"] else set()
        for n, c in counts.items():
            if c >= args.min_count and n:
                tasks.append((rec["SubjectEntity"], surface[n], c / m, n in gold_aliases))
    print(f"{args.relation}: {len(tasks)} candidates from {len(dumps)} rows "
          f"({sum(t[3] for t in tasks)} true)", file=sys.stderr)

    client = Client(base_url=args.base_url, model=args.model)
    client.wait_until_ready()
    q = QUESTION[args.relation]

    from threading import Lock
    diag = {"err": 0, "unparsed": 0, "ok": 0}
    diag_lock = Lock()
    first_errors: list[str] = []
    first_raw: list[str] = []

    def verify(subject: str, cand: str, seed: int) -> bool | None:
        try:
            text = client.complete(
                prompt=prefix + q.format(c=cand, s=subject),
                temperature=0.7, max_tokens=3, seed=4200 + seed, stop=["\n"])
        except ServerUnavailable:
            raise
        except Exception as e:
            with diag_lock:
                diag["err"] += 1
                if len(first_errors) < 3:
                    first_errors.append(repr(e)[:200])
            return None
        word = re.search(r"[a-z]+", str(text).lower())
        with diag_lock:
            if len(first_raw) < 5:
                first_raw.append(repr(text)[:60])
        if word and word.group(0) in ("yes", "no"):
            with diag_lock:
                diag["ok"] += 1
            return word.group(0) == "yes"
        with diag_lock:
            diag["unparsed"] += 1
        return None

    votes = defaultdict(lambda: [0, 0])
    jobs = [(s, c, i) for s, c, _, _ in tasks for i in range(args.samples)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(verify, s, c, i): (s, c) for s, c, i in jobs}
        for f in tqdm(as_completed(futs), total=len(futs), desc="verify"):
            key = futs[f]
            try:
                r = f.result()
            except ServerUnavailable as e:
                for pf in futs:
                    pf.cancel()
                raise SystemExit(f"FATAL: {e}")
            if r is not None:
                votes[key][0] += int(r)
                votes[key][1] += 1
    client.close()
    print(f"calls: ok={diag['ok']} unparsed={diag['unparsed']} err={diag['err']}")
    if first_errors:
        print("first errors:", *first_errors, sep="\n  ")
    print("first raw completions:", *first_raw, sep="\n  ")
    if diag["ok"] == 0:
        raise SystemExit("no call produced yes/no -- probe invalid, fix before trusting output")

    scored = []
    for s, c, sup, is_true in tasks:
        y, a = votes[(s, c)]
        scored.append({"subject": s, "candidate": c, "support": sup,
                       "true": is_true, "yes": y, "asks": a,
                       "score": (y / a) if a else 0.5})

    import random as rnd
    rnd.seed(0)
    pos = [x["score"] for x in scored if x["true"]]
    neg = [x["score"] for x in scored if not x["true"]]
    wins = float("nan")
    if pos and neg:
        total = 0.0
        for _ in range(20000):
            p, n = rnd.choice(pos), rnd.choice(neg)
            total += 1.0 if p > n else (0.5 if p == n else 0.0)
        wins = total / 20000
    print(f"verification AUC (yes-rate): {wins:.3f}")

    by_row = defaultdict(list)
    for x in scored:
        by_row[x["subject"]].append(x)
    for v in (0.5, 0.625, 0.75, 0.875, 1.0):
        f1s = []
        for rec in dumps:
            g = gold.get((rec["SubjectEntity"], args.relation))
            if g is None:
                continue
            ents = alias_sets(g)
            pred = [x for x in by_row.get(rec["SubjectEntity"], []) if x["score"] >= v]
            S = len(pred)
            T = sum(1 for al in ents if any(norm(x["candidate"]) in al for x in pred))
            G = len(ents)
            if G == 0 and S == 0:
                f1s.append(1.0)
            elif G == 0 or S == 0:
                f1s.append(0.0)
            else:
                f1s.append(2 * T / (S + G))
        print(f"  emit if yes-rate>={v:.3f}: F1={sum(f1s)/len(f1s):.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(scored, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"wrote {args.out}")

REL = "personHasCityOfDeath"

LINE = "As of 1 July 2026, {s} is {v}."

def _existence_prefix(train_rows: list[dict], shots: int) -> str:
    pool = [r for r in train_rows if r["Relation"] == REL]
    dead = [r for r in pool if r["ObjectEntities"]]
    alive = [r for r in pool if not r["ObjectEntities"]]
    rng = random.Random(31)
    chosen = ([(r, "dead") for r in rng.sample(dead, shots // 2)] +
              [(r, "alive") for r in rng.sample(alive, shots // 2)])
    rng.shuffle(chosen)
    lines = [LINE.format(s=r["SubjectEntity"], v=v) for r, v in chosen]
    return "\n".join(lines) + "\n" + LINE.split("{v}")[0].replace("{s}", "{subject}").rstrip()

def existence_check_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [r for r in read(args.gold) if r["Relation"] == REL]
    prefix = _existence_prefix(read(args.train), args.shots)
    print(f"--- prefix ---\n{prefix}\n---", file=sys.stderr)

    client = Client(base_url=args.base_url, model=args.model)
    client.wait_until_ready()

    def ask(subject: str, seed: int) -> bool | None:
        try:
            text = client.complete(
                prompt=prefix.format(subject=subject),
                temperature=0.7, max_tokens=3, seed=5100 + seed, stop=["\n"])
        except ServerUnavailable:
            raise
        except Exception:
            return None
        w = re.search(r"[a-z]+", str(text).lower())
        if w and w.group(0) in ("dead", "alive", "deceased", "living"):
            return w.group(0) in ("dead", "deceased")
        return None

    votes = defaultdict(lambda: [0, 0])
    jobs = [(r["SubjectEntity"], i) for r in rows for i in range(args.samples)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ask, s, i): s for s, i in jobs}
        for f in tqdm(as_completed(futs), total=len(futs), desc="alive"):
            s = futs[f]
            try:
                r = f.result()
            except ServerUnavailable as e:
                for pf in futs:
                    pf.cancel()
                raise SystemExit(f"FATAL: {e}")
            if r is not None:
                votes[s][0] += int(r)
                votes[s][1] += 1

    client.close()
    parsed = sum(1 for v in votes.values() if v[1])
    print(f"parsed subjects: {parsed}/{len(rows)}")
    if parsed == 0:
        raise SystemExit("no completions parsed -- probe invalid")

    scored = []
    for r in rows:
        y, a = votes[r["SubjectEntity"]]
        scored.append({"subject": r["SubjectEntity"],
                       "dead_rate": (y / a) if a else 0.5,
                       "gold_dead": bool(r["ObjectEntities"])})

    import random as rnd
    rnd.seed(0)
    pos = [x["dead_rate"] for x in scored if x["gold_dead"]]
    neg = [x["dead_rate"] for x in scored if not x["gold_dead"]]
    if pos and neg:
        tot = 0.0
        for _ in range(20000):
            p, n = rnd.choice(pos), rnd.choice(neg)
            tot += 1.0 if p > n else (0.5 if p == n else 0.0)
        print(f"existence-check AUC (dead-rate vs gold non-empty): {tot/20000:.3f}")
    Path(args.out).write_text(json.dumps(scored, ensure_ascii=False),
                              encoding="utf-8")
    print(f"wrote {args.out}")

AWARD = "awardWonBy"

DEATH = "personHasCityOfDeath"

def merge_variants(names: list[str]) -> list[str]:
    kept: list[str] = []
    toks = {n: normalize(n).split() for n in names}
    for name in sorted(names, key=lambda n: -len(toks[n])):
        t = toks[name]
        if not t:
            continue
        dup = False
        for other in kept:
            o = toks[other]
            if not o or t == o:
                dup = dup or t == o
                continue
            if t[0] == o[0] and t[-1] == o[-1] and set(t) <= set(o):
                dup = True
                break
        if not dup:
            kept.append(name)
    return kept

def alias_pairs(*gold_files: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for path in gold_files:
        for row in read(path):
            for aliases in row.get("ObjectEntities") or []:
                if not isinstance(aliases, list):
                    continue
                forms = sorted({normalize(a) for a in aliases
                                if isinstance(a, str) and a.strip()})
                pairs.update(itertools.combinations(forms, 2))
    return pairs


def dedup(preds: list[str], pairs: set[tuple[str, str]]) -> list[str]:
    kept: list[str] = []
    for p in preds:
        if not isinstance(p, str):
            continue
        k = normalize(p)
        if any(tuple(sorted((k, normalize(q)))) in pairs for q in kept):
            continue
        kept.append(p)
    return kept


def apply_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="completion_system output jsonl")
    ap.add_argument("--ver", default=None, help="verification_rescue --out json")
    ap.add_argument("--existence", default=None, help="postprocess.py existence-check --out json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", required=True,
                    help="training split; alias lists for the dedup are read from it")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--samples", type=int, default=64,
                    help="sampling budget the dump was produced at; min-count/samples "
                         "must stay below the fitted awardWonBy threshold or the "
                         "rescue has no band to work in and does nothing")
    ap.add_argument("--yes-rate", type=float, default=0.75)
    args = ap.parse_args()

    rows = read(args.pred)
    stats = {"rescued": 0, "merged": 0, "gated": 0, "deduped": 0}

    if args.ver:
        floor = args.min_count / args.samples
        rescue: dict[str, list[str]] = {}
        for r in json.loads(Path(args.ver).read_text(encoding="utf-8")):
            if r["support"] >= floor and r["score"] >= args.yes_rate:
                rescue.setdefault(r["subject"], []).append(r["candidate"])
        for row in rows:
            if row["Relation"] != AWARD:
                continue
            have = {normalize(o) for o in row["ObjectEntities"]}
            added = [c for c in rescue.get(row["SubjectEntity"], [])
                     if normalize(c) not in have]
            merged = merge_variants(row["ObjectEntities"] + added)
            stats["rescued"] += len(added)
            stats["merged"] += len(row["ObjectEntities"]) + len(added) - len(merged)
            row["ObjectEntities"] = merged

    if args.existence:
        alive = {a["subject"]: a["dead_rate"]
                 for a in json.loads(Path(args.existence).read_text(encoding="utf-8"))}
        for row in rows:
            if row["Relation"] != DEATH:
                continue
            if alive.get(row["SubjectEntity"], 1.0) == 0.0 and row["ObjectEntities"]:
                row["ObjectEntities"] = []
                stats["gated"] += 1

    pairs = alias_pairs(args.train)
    for row in rows:
        before = row.get("ObjectEntities") or []
        after = dedup(before, pairs)
        stats["deduped"] += len(before) - len(after)
        row["ObjectEntities"] = after

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: rescued {stats['rescued']} award candidates, "
          f"merged {stats['merged']} duplicate names, gated {stats['gated']} death rows, "
          f"dropped {stats['deduped']} alias duplicates")

def main() -> None:
    import sys
    modes = {"rescue": rescue_main, "existence-check": existence_check_main, "apply": apply_main}
    if len(sys.argv) < 2 or sys.argv[1] not in modes:
        raise SystemExit(f"usage: postprocess.py {{{'|'.join(modes)}}} [options]")
    modes[sys.argv.pop(1)]()


if __name__ == "__main__":
    main()
