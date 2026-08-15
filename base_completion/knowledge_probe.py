# The knowledge probe: voted and ceiling, per checkpoint and question form.
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from client import Client, ServerUnavailable, extract_value
from aggregation import (
    AREA_FORMS,
    CAPACITY_FORMS,
    QuestionForm,
    cluster_vote,
    parse_number,
    within_tolerance,
)
import random
from scorer import load_official
from aggregation import SET_SPECS, aggregate

FORMS: dict[str, tuple[QuestionForm, ...]] = {
    "hasArea": AREA_FORMS,
    "hasCapacity": CAPACITY_FORMS,
}

def load_rows(path: str, relation: str, limit: int) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if record.get("Relation") != relation:
                continue

            objects = record.get("ObjectEntities") or []
            if not objects:
                continue

            first = objects[0]
            first = first[0] if isinstance(first, list) and first else first

            try:
                gold = float(str(first).replace(",", ""))
            except (ValueError, TypeError):
                continue

            rows.append((record["SubjectEntity"], gold))

    return rows[:limit] if limit else rows

def sample_one(
    client: Client,
    form: QuestionForm,
    subject: str,
    *,
    index: int,
    temperature: float,
    thinking: bool,
    max_tokens: int,
) -> float | None:
    try:
        if form.freeform:
            text = client.complete(
                prompt=form.render(subject),
                temperature=temperature,
                max_tokens=48,
                seed=1234 + index,
            )
            value = parse_number(text, detect_units=True)
        else:
            text = client.chat(
                system=form.system,
                prompt=form.render(subject),
                temperature=temperature,
                max_tokens=max_tokens,
                seed=1234 + index,
                thinking=thinking,
            )
            raw = extract_value(text)
            value = parse_number(
                str(raw), detect_units=getattr(form, "parse_units", False)
            )
    except ServerUnavailable:
        raise
    except Exception:
        return None

    if value is None:
        return None

    return value * form.scale

def numeric_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--relation", default="hasArea")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Hybrid/Thinking checkpoints only.",
    )
    parser.add_argument(
        "--system-mode",
        default="auto",
        choices=["auto", "system", "merge"],
        help="'merge' folds the system prompt into the user turn, for chat "
             "templates that reject a system role (Gemma).",
    )
    parser.add_argument(
        "--train",
        default=None,
        help="Train gold jsonl. Enables the LAMA-style 'fewshot' completion "
             "form: five real (subject, value) exemplars as plain text, "
             "then the target sentence, via the raw completions API.",
    )
    parser.add_argument(
        "--forms",
        default="",
        help="Comma-separated allowlist. For base models without a chat "
             "template, restrict to the completion channels: fewshot,freeform",
    )
    parser.add_argument("--out", default=None, help="Raw values as JSON.")
    args = parser.parse_args()

    forms = FORMS.get(args.relation)
    if not forms:
        raise SystemExit(f"no forms for relation {args.relation}")

    rows = load_rows(args.gold, args.relation, args.limit)
    if not rows:
        raise SystemExit(f"no numeric gold rows for {args.relation}")

    fewshot_prefix = None
    if args.train:
        SENT = {
            "hasArea": "The total area of {s} is {v} square kilometres.",
            "hasCapacity": "The seating capacity of {s} is {v}.",
        }[args.relation]
        import random as _random
        exemplars = load_rows(args.train, args.relation, 0)
        exemplars.sort(key=lambda r: r[1])
        step = max(1, len(exemplars) // 5)
        picked = exemplars[::step][:5]
        _random.Random(7).shuffle(picked)
        fewshot_prefix = "\n".join(
            SENT.format(s=s_, v=(int(v) if v == int(v) else v))
            for s_, v in picked
        ) + "\n" + SENT.split(" is ")[0].replace("{s}", "{subject}") + " is"

    client = Client(
        base_url=args.base_url,
        model=args.model,
        system_mode=args.system_mode,
    )
    print("waiting for vLLM ...", file=sys.stderr)
    client.wait_until_ready()

    print(
        f"{len(rows)} rows x {len(forms)} forms x {args.samples} samples "
        f"= {len(rows) * len(forms) * args.samples} calls",
        file=sys.stderr,
    )

    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    all_forms = list(forms)
    if fewshot_prefix is not None:
        all_forms.append(QuestionForm(
            name="fewshot",
            system="",
            freeform=True,
            parse_units=True,
            template=fewshot_prefix,
        ))

    if args.forms:
        wanted = {n.strip() for n in args.forms.split(",") if n.strip()}
        all_forms = [e for e in all_forms if e.name in wanted]
        if not all_forms:
            raise SystemExit(f"no forms match {sorted(wanted)}")

    jobs = [
        (subject, form, index)
        for subject, _ in rows
        for form in all_forms
        for index in range(args.samples)
    ]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                sample_one,
                client,
                form,
                subject,
                index=index,
                temperature=args.temperature,
                thinking=args.thinking,
                max_tokens=args.max_tokens,
            ): (subject, form.name)
            for subject, form, index in jobs
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="calls"):
            subject, name = futures[future]

            try:
                value = future.result()
            except ServerUnavailable as error:
                for pending in futures:
                    pending.cancel()
                raise SystemExit(f"\nFATAL: {error}\nprobe results discarded")

            if value is not None:
                values[subject][name].append(value)

    client.close()

    gold_by_subject = dict(rows)

    header = (
        f"{'form':<12} {'hit@1':>7} {'hit@any':>8} {'cluster':>8} "
        f"{'parsed':>7}"
    )
    print(f"\nrelation: {args.relation}   n={len(rows)}   "
          f"samples={args.samples}   thinking={args.thinking}")
    print(header)
    print("-" * len(header))

    for form in all_forms:
        hit1 = hit_any = cluster_hit = parsed = 0

        for subject, gold in rows:
            samples = values[subject][form.name]
            parsed += bool(samples)

            if samples:
                hit1 += within_tolerance(samples[0], gold)
                hit_any += any(within_tolerance(v, gold) for v in samples)
                cluster_hit += within_tolerance(cluster_vote(samples)[0], gold)

        n = len(rows)
        print(
            f"{form.name:<12} {hit1 / n:>7.3f} {hit_any / n:>8.3f} "
            f"{cluster_hit / n:>8.3f} {parsed / n:>7.3f}"
        )

    union_any = union_cluster = 0
    for subject, gold in rows:
        pooled = [v for group in values[subject].values() for v in group]
        union_any += any(within_tolerance(v, gold) for v in pooled)
        union_cluster += within_tolerance(cluster_vote(pooled)[0], gold)

    n = len(rows)
    print("-" * len(header))
    print(f"{'UNION':<12} {'':>7} {union_any / n:>8.3f} {union_cluster / n:>8.3f}")
    print(
        "\nhit@any is the ceiling for any aggregator over that form; UNION\n"
        "hit@any is the ceiling for an ensemble over all of them."
    )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "relation": args.relation,
                    "thinking": args.thinking,
                    "model": args.model,
                    "gold": gold_by_subject,
                    "values": {s: dict(v) for s, v in values.items()},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")

TEMPLATES = {
    "countryLandBordersCountry":
        "Countries sharing a land border with {s}: {v}",
    "companyTradesAtStockExchange":
        "Stock exchanges where {s} is listed: {v}",
    "personHasCityOfDeath":
        "City where {s} died (or 'none' if alive as of July 2026): {v}",
    "awardWonBy":
        "Recipients of {s}: {v}",
}

_SET_EMPTY_WORD = "none"

def _set_gold_labels(row: dict) -> list[str]:
    return [g[0] if isinstance(g, list) else g for g in row["ObjectEntities"]]

def _set_prefix(relation: str, train_rows: list[dict], shots: int) -> str:
    template = TEMPLATES[relation]
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
        labels = _set_gold_labels(r)
        value = ", ".join(labels[:12]) if labels else _SET_EMPTY_WORD
        lines.append(template.format(s=r["SubjectEntity"], v=value))
    target = template.format(s="{subject}", v="").rstrip()
    return "\n".join(lines) + "\n" + target

def parse_line(text: str) -> list[str]:
    line = str(text or "").strip().split("\n")[0].strip()
    if not line or line.lower().startswith(_SET_EMPTY_WORD):
        return []
    parts = [p.strip(" .;") for p in line.split(",")]
    return [p for p in parts if p and len(p) < 80]

def sets_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="val jsonl")
    parser.add_argument("--train", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--template", default=None,
        help="Override the completion template; must contain {s} and {v}.")
    parser.add_argument(
        "--empty-word", default=None,
        help="Override the token written for empty gold sets.")
    args = parser.parse_args()

    if args.template:
        TEMPLATES[args.relation] = args.template
    global _SET_EMPTY_WORD
    if args.empty_word:
        _SET_EMPTY_WORD = args.empty_word

    official = load_official()
    gold_rows = [r for r in official.read_jsonl_file(args.gold)
                 if r["Relation"] == args.relation]
    if args.limit:
        gold_rows = gold_rows[: args.limit]
    train_rows = official.read_jsonl_file(args.train)

    prefix = _set_prefix(args.relation, train_rows, args.shots)
    print(f"--- few-shot prefix ({args.relation}) ---\n{prefix}\n---",
          file=sys.stderr)

    client = Client(base_url=args.base_url, model=args.model)
    client.wait_until_ready()

    def sample(subject: str, seed: int) -> list[str]:
        try:
            text = client.complete(
                prompt=prefix.format(subject=subject),
                temperature=0.7, max_tokens=args.max_tokens,
                seed=1300 + seed, stop=["\n"],
            )
        except ServerUnavailable:
            raise
        except Exception:
            return None
        return parse_line(text)

    collected: dict[str, list] = {r["SubjectEntity"]: [] for r in gold_rows}
    jobs = [(r["SubjectEntity"], i) for r in gold_rows
            for i in range(args.samples)]

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(sample, s, i): s for s, i in jobs}
        for f in tqdm(as_completed(futs), total=len(futs), desc="calls"):
            s = futs[f]
            try:
                answers = f.result()
            except ServerUnavailable as e:
                for pf in futs: pf.cancel()
                raise SystemExit(f"FATAL: {e}")
            if answers is not None:
                collected[s].append(answers)

    client.close()

    spec = SET_SPECS[args.relation]
    print(f"{args.relation}: n={len(gold_rows)}, "
          f"mean samples/row {sum(len(v) for v in collected.values())/len(gold_rows):.1f}")

    best = (0.0, None)
    for support in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        preds = [{"SubjectEntity": r["SubjectEntity"],
                  "Relation": args.relation,
                  "ObjectEntities": aggregate(
                      collected[r["SubjectEntity"]], spec,
                      min_support=support)}
                 for r in gold_rows]
        scores = official.evaluate_per_sr_pair(
            preds, gold_rows, official.RELATION_TYPE, tolerance=0.05)
        f1 = sum(x["f1"] for x in scores) / len(scores)
        marker = ""
        if f1 > best[0]:
            best = (f1, support); marker = "  <-"
        print(f"  support={support:.1f}  F1={f1:.4f}{marker}")

    print(f"BEST {args.relation}: F1={best[0]:.4f} at support={best[1]}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"relation": args.relation,
             "samples": {s: v for s, v in collected.items()}},
            ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")

def main() -> None:
    import sys
    modes = {"numeric": numeric_main, "sets": sets_main}
    if len(sys.argv) < 2 or sys.argv[1] not in modes:
        raise SystemExit(f"usage: knowledge_probe.py {{{'|'.join(modes)}}} [options]")
    modes[sys.argv.pop(1)]()


if __name__ == "__main__":
    main()
