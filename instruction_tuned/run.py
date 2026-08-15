# Entry point: predict, fit, score, ablate, sweep, validate, package.
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

from .decode import Calibrator, reliability
from .data import RELATIONS, Row, load_split, write_predictions
from .decode import EXCLUSIVE, INDEPENDENT, decode, geometric_missing_mass
from .group import Candidate, score_candidates_against_gold
from .decode import format_report, oracle_scores, score_all
from .propose import RowPrediction, Solver, SolverConfig, _EXCLUSIVE_RELATIONS
from .clients import ResponseCache, check_param_budget

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
CACHE_PATH = RUNS_DIR / "cache.sqlite"

DEFAULT_MODEL = "google/gemma-3-27b-it"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _select(rows: Sequence[Row], relations: Sequence[str] | None, limit: int) -> list[Row]:
    chosen = list(relations or RELATIONS)
    out: list[Row] = []
    for relation in chosen:
        subset = [r for r in rows if r.relation == relation]
        out.extend(subset[:limit] if limit else subset)
    return out


def _load_checkpoint(path: Path) -> dict[tuple[str, str], RowPrediction]:
    if not path.exists():
        return {}
    done: dict[tuple[str, str], RowPrediction] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            prediction = RowPrediction(**record)
            done[prediction.key] = prediction
    return done


def _client_factory(args: argparse.Namespace):
    from .clients import VLLMClient

    def build(model: str, **kwargs):
        return VLLMClient(
            model=model,
            base_url=args.base_url,
            max_concurrency=args.concurrency,
            **kwargs,
        )

    return build


def _build_proposers(mechanism: str, client, iterations: int, base=None) -> dict:
    if mechanism == "sampling":
        return {}

    if mechanism == "structured":
        from .propose import StructuredProposer

        proposer = StructuredProposer(client)
        return {relation: proposer for relation in RELATIONS}

    from .propose import SetExpansionSearch

    if mechanism in ("expansion", "auto"):
        return {"awardWonBy": SetExpansionSearch(client, iterations=iterations, base=base)}

    raise ValueError(f"unknown proposer {mechanism!r}")


async def cmd_predict(args: argparse.Namespace) -> int:
    rows = load_split(args.split)
    train_rows = load_split("train")
    targets = _select(rows, args.relations, args.limit)

    models = [args.model] + ([args.verifier] if args.verifier else [])
    ok, verdict = check_param_budget(models)
    print(f"model budget: {verdict}", file=sys.stderr)
    if not ok and not args.allow_over_budget:
        print(
            "  refusing to run: this configuration cannot be submitted. Pass "
            "--allow-over-budget to run it anyway as a development-only reference.",
            file=sys.stderr,
        )
        return 2

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.split}-{args.model.split('/')[-1]}"
    checkpoint_path = RUNS_DIR / f"{tag}.checkpoint.jsonl"
    output_path = Path(args.out) if args.out else RUNS_DIR / f"{tag}.jsonl"

    done = {} if args.fresh else _load_checkpoint(checkpoint_path)
    todo = [r for r in targets if r.key not in done]
    print(
        f"{len(targets)} rows selected, {len(done)} already done, {len(todo)} to run",
        file=sys.stderr,
    )

    cache = None if args.no_cache else ResponseCache(str(CACHE_PATH))
    make_client = _client_factory(args)
    client = make_client(args.model, temperature=args.temperature, cache=cache)
    verifier = make_client(args.verifier, cache=cache) if args.verifier else None

    calibrator = (
        Calibrator.load(args.calibrator)
        if args.calibrator and Path(args.calibrator).exists()
        else Calibrator.default()
    )
    if args.calibrator and not Path(args.calibrator).exists():
        print(
            f"  no calibrator at {args.calibrator}; using the uncalibrated blend",
            file=sys.stderr,
        )

    missing_mass_cap = args.missing_mass_cap
    settings_path = Path(args.calibrator).with_suffix(".settings.json")
    if missing_mass_cap is None and settings_path.exists():
        swept = json.loads(settings_path.read_text(encoding="utf-8"))
        missing_mass_cap = float(swept.get("missing_mass_cap", 1.0))
        print(
            f"  using swept settings from {settings_path.name}: "
            f"missing_mass_cap={missing_mass_cap}",
            file=sys.stderr,
        )
    config = SolverConfig(
        n_samples=args.samples,
        temperature=args.temperature,
        n_shots=args.shots,
        verify=not args.no_verify,
        estimate_count=not args.no_count,
        missing_mass_cap=1.0 if missing_mass_cap is None else missing_mass_cap,
        seed=args.seed,
    )
    sampling_base = Solver(client, train_rows, config, calibrator, verifier)
    proposers = _build_proposers(
        args.proposer, client, args.expansion_iterations, base=sampling_base
    )
    if proposers:
        print(
            f"proposal mechanism: {args.proposer} "
            f"(applied to {sorted(proposers)})",
            file=sys.stderr,
        )
    solver = Solver(client, train_rows, config, calibrator, verifier, proposers)

    started = time.time()
    checkpoint = open(checkpoint_path, "a", encoding="utf-8")
    completed = 0
    try:
        semaphore = asyncio.Semaphore(args.row_concurrency)

        async def run_row(row: Row) -> None:
            nonlocal completed
            async with semaphore:
                prediction = await solver.solve_row(row)
            done[prediction.key] = prediction
            checkpoint.write(json.dumps(prediction.__dict__, ensure_ascii=False) + "\n")
            checkpoint.flush()
            completed += 1
            if completed % 10 == 0 or completed == len(todo):
                elapsed = time.time() - started
                rate = completed / elapsed if elapsed else 0
                remaining = (len(todo) - completed) / rate if rate else 0
                print(
                    f"  {completed}/{len(todo)} rows  {elapsed:6.0f}s elapsed  "
                    f"~{remaining:5.0f}s left  cache_hits={client.n_cache_hits}",
                    file=sys.stderr,
                )

        await asyncio.gather(*(run_row(r) for r in todo))
    finally:
        checkpoint.close()
        await client.aclose()
        if verifier:
            await verifier.aclose()

    predictions = {key: p.objects for key, p in done.items()}
    write_predictions(output_path, predictions, targets)
    print(f"\nwrote {output_path}", file=sys.stderr)
    print(
        f"tokens: in={client.usage.tokens_in:,} out={client.usage.tokens_out:,}  "
        f"provider_calls={client.n_provider_calls}  cache_hits={client.n_cache_hits}",
        file=sys.stderr,
    )

    if args.split != "test":
        _, by_relation, overall = score_all(targets, predictions)
        print()
        print(format_report(by_relation, overall))
        print("\n" + _abstention_breakdown(list(done.values())))
        pools = {p.key: [c["label"] for c in p.candidates] for p in done.values()}
        print("\noracle F1 over the proposed candidate pool (ranking ceiling):")
        for relation, value in sorted(oracle_scores(targets, pools).items()):
            print(f"  {relation:32s} {value:.3f}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    rows = {r.key: r for r in load_split(args.split)}
    predictions = _load_checkpoint(Path(args.checkpoint))
    if not predictions:
        print(f"no checkpoint rows at {args.checkpoint}", file=sys.stderr)
        return 1

    calibrator = Calibrator()
    for relation in RELATIONS:
        features: list[dict[str, float]] = []
        labels: list[int] = []
        for key, prediction in predictions.items():
            if prediction.relation != relation or key not in rows:
                continue
            row = rows[key]
            candidates = [
                Candidate(
                    label=c["label"],
                    support=c["support"],
                    n_samples=c["n_samples"],
                    value=c.get("value"),
                    features=c.get("features", {}),
                )
                for c in prediction.candidates
            ]
            if not candidates:
                continue
            hits = score_candidates_against_gold(
                candidates, row.gold, numeric=RELATIONS[relation].is_numeric
            )
            for candidate, hit in zip(candidates, hits):
                merged = dict(candidate.features)
                merged["recurrence"] = candidate.smoothed_frequency
                features.append(merged)
                labels.append(hit)

        if features:
            calibrator.fit_relation(relation, features, labels, l2=args.l2)
            probabilities = [calibrator.probability(relation, f) for f in features]
            print(f"\n{relation}  (n={len(features)}, positives={sum(labels)})")
            print(reliability(probabilities, labels))

    print("\nfitted coefficients:")
    print(calibrator.report())
    path = calibrator.save(args.out)
    print(f"\nwrote {path}", file=sys.stderr)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    rows = load_split(args.split)
    predictions: dict[tuple[str, str], list[str]] = {}
    with open(args.predictions, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                predictions[(record["SubjectEntity"], record["Relation"])] = record[
                    "ObjectEntities"
                ]
    covered = [r for r in rows if r.key in predictions]
    scores, by_relation, overall = score_all(covered, predictions)
    print(format_report(by_relation, overall))

    if args.worst:
        print(f"\nworst {args.worst} rows:")
        for score in sorted(scores, key=lambda s: s.f1)[: args.worst]:
            row = next(r for r in covered if r.key == (score.subject, score.relation))
            print(
                f"  [{score.f1:.2f}] {score.relation} / {score.subject}\n"
                f"        pred: {predictions[row.key]}\n"
                f"        gold: {row.gold_labels[:12]}"
            )
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    rows = {r.key: r for r in load_split(args.split)}
    predictions = _load_checkpoint(Path(args.checkpoint))
    if not predictions:
        print(f"no checkpoint rows at {args.checkpoint}", file=sys.stderr)
        return 1

    calibrator = (
        Calibrator.load(args.calibrator)
        if args.calibrator and Path(args.calibrator).exists()
        else Calibrator.default()
    )
    covered = [rows[k] for k in predictions if k in rows]

    def build(condition: str) -> dict[tuple[str, str], list[str]]:
        out: dict[tuple[str, str], list[str]] = {}
        for key, prediction in predictions.items():
            if key not in rows:
                continue
            spec = RELATIONS[prediction.relation]
            candidates = [
                Candidate(
                    label=c["label"],
                    support=c["support"],
                    n_samples=c["n_samples"],
                    features=c.get("features", {}),
                )
                for c in prediction.candidates
            ]
            if not candidates:
                out[key] = []
                continue

            candidates.sort(key=lambda c: -c.support)
            for rank, candidate in enumerate(candidates):
                merged = dict(candidate.features)
                merged["recurrence"] = candidate.smoothed_frequency
                merged["rank_fraction"] = 1.0 / (1.0 + rank)
                merged["pool_fraction"] = 1.0 / (1.0 + len(candidates))
                if condition == "freq-only":
                    merged["verification"] = 0.5
                elif condition == "verify-only":
                    merged["recurrence"] = 0.5
                candidate.prob = calibrator.probability(prediction.relation, merged)
            candidates.sort(key=lambda c: -c.prob)

            if condition == "top1":
                out[key] = [candidates[0].label]
            elif condition == "majority":
                picked = [c.label for c in candidates if c.frequency >= 0.5]
                out[key] = picked or [candidates[0].label]
            elif condition == "majority-abstain":
                out[key] = [c.label for c in candidates if c.frequency >= 0.5]
            elif condition == "all":
                out[key] = [c.label for c in candidates]
            else:
                mode = (
                    EXCLUSIVE if prediction.relation in _EXCLUSIVE_RELATIONS else INDEPENDENT
                )
                missing = None
                if mode == INDEPENDENT and prediction.count_estimate:
                    present = sum(c.prob for c in candidates)
                    missing = geometric_missing_mass(
                        max(0.0, prediction.count_estimate - present)
                    )
                result = decode(
                    [c.label for c in candidates],
                    [c.prob for c in candidates],
                    mode=mode,
                    missing_mass=missing,
                    gold_is_nonempty=not spec.allows_empty,
                    max_k=spec.max_candidates,
                )
                out[key] = result.selected
        return out

    conditions = [
        "top1",
        "majority",
        "majority-abstain",
        "all",
        "freq-only",
        "verify-only",
        "expected-f1",
    ]
    header = (
        f"{'condition':18s} {'macro-f1':>9s} {'macro-p':>8s} {'macro-r':>8s} "
        f"{'#preds':>7s} {'empty':>7s}"
    )
    print(header)
    print("-" * len(header))
    results = {}
    for condition in conditions:
        built = build(condition)
        _, by_relation, overall = score_all(covered, built)
        results[condition] = by_relation
        print(
            f"{condition:18s} {overall.macro_f1:9.3f} {overall.macro_p:8.3f} "
            f"{overall.macro_r:8.3f} {overall.avg_preds:7.2f} {overall.n_empty_pred:7d}"
        )

    print("\nper-relation macro-F1:")
    header = f"{'relation':32s}" + "".join(f"{c[:17]:>18s}" for c in conditions)
    print(header)
    print("-" * len(header))
    for relation in sorted({p.relation for p in predictions.values()}):
        cells = "".join(
            f"{results[c].get(relation).macro_f1:18.3f}"
            if results[c].get(relation)
            else f"{'-':>18s}"
            for c in conditions
        )
        print(f"{relation:32s}{cells}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from .calibrate import fit_calibrator, format_sweep, load_checkpoint, sweep

    results = sweep(args.checkpoint, args.split)
    print(format_sweep(results, top=args.top))
    best = results[0]
    print(f"\nbest: {best.label()}  macro-f1={best.macro_f1:.4f}")
    print(f"spread across the grid: {results[-1].macro_f1:.4f} .. {best.macro_f1:.4f}")

    rows = {r.key: r for r in load_split(args.split)}
    records = [
        r for r in load_checkpoint(args.checkpoint) if (r["subject"], r["relation"]) in rows
    ]
    calibrator = fit_calibrator(
        records, rows, l2=best.l2, calibration_floor=best.calibration_floor
    )
    path = calibrator.save(args.out)
    settings = Path(args.out).with_suffix(".settings.json")
    settings.write_text(
        json.dumps(
            {
                "l2": best.l2,
                "calibration_floor": best.calibration_floor,
                "missing_mass_cap": best.missing_mass_cap,
                "use_verify": best.use_verify,
                "train_macro_f1": best.macro_f1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {path} and {settings}", file=sys.stderr)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    rows = load_split(args.split)
    expected = {r.key for r in rows}
    seen: dict[tuple[str, str], list] = {}
    problems: list[str] = []

    with open(args.predictions, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"line {number}: not valid JSON ({exc})")
                continue
            missing_fields = {"SubjectEntity", "Relation", "ObjectEntities"} - set(record)
            if missing_fields:
                problems.append(f"line {number}: missing {sorted(missing_fields)}")
                continue
            objects = record["ObjectEntities"]
            if not isinstance(objects, list):
                problems.append(f"line {number}: ObjectEntities is not a list")
                continue
            bad = [o for o in objects if not isinstance(o, str)]
            if bad:
                problems.append(f"line {number}: non-string objects {bad[:3]}")
            key = (record["SubjectEntity"], record["Relation"])
            if key in seen:
                problems.append(f"line {number}: duplicate row {key}")
            seen[key] = objects

    absent = expected - set(seen)
    extra = set(seen) - expected
    if absent:
        problems.append(f"{len(absent)} rows missing, e.g. {sorted(absent)[:3]}")
    if extra:
        problems.append(f"{len(extra)} unexpected rows, e.g. {sorted(extra)[:3]}")

    print(f"{len(seen)} rows in file, {len(expected)} expected for split {args.split!r}")
    counts: dict[str, int] = {}
    empties: dict[str, int] = {}
    for (_, relation), objects in seen.items():
        counts[relation] = counts.get(relation, 0) + 1
        empties[relation] = empties.get(relation, 0) + (1 if not objects else 0)
    for relation in sorted(counts):
        total = sum(len(o) for k, o in seen.items() if k[1] == relation)
        print(
            f"  {relation:32s} {counts[relation]:4d} rows  "
            f"{empties[relation]:4d} empty  {total / counts[relation]:6.2f} objects/row"
        )

    if problems:
        print("\nPROBLEMS:")
        for problem in problems[:20]:
            print(f"  - {problem}")
        return 1
    print("\nvalid submission file")
    return 0


def cmd_package(args: argparse.Namespace) -> int:
    import zipfile

    validation = cmd_validate(args)
    if validation != 0:
        print("\nrefusing to package an invalid prediction file", file=sys.stderr)
        return validation

    source = Path(args.predictions)
    out = Path(args.out or (RUNS_DIR / f"submission-{args.split}.zip"))
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(source, arcname="predictions.jsonl")

    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
    print(f"\nwrote {out}")
    print(f"  contents: {names}")
    print(f"  size: {out.stat().st_size:,} bytes")
    if names != ["predictions.jsonl"]:
        print("  WARNING: archive layout is not what the competition expects")
        return 1
    return 0


def _abstention_breakdown(predictions) -> str:
    no_candidates = sum(1 for p in predictions if not p.candidates and not p.objects)
    decoder = sum(1 for p in predictions if p.candidates and not p.objects)
    return (
        f"empty predictions: {no_candidates} from an empty candidate pool "
        f"(the prompt declined), {decoder} from the decoder choosing k=0"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lmkbc.run", description="Entry point: predict, fit, score, ablate, sweep, validate, package.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="run the pipeline over a split")
    predict.add_argument("--split", default="val", choices=["train", "val", "test"])
    predict.add_argument("--model", default=DEFAULT_MODEL)
    predict.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM server root, used with --backend vllm",
    )
    predict.add_argument(
        "--verifier", default=None, help="separate model for the verification stage"
    )
    predict.add_argument(
        "--proposer",
        default="sampling",
        choices=["sampling", "structured", "expansion", "auto"],
        help="stage-1 mechanism; stages 2-4 are identical across all of them. "
             "'auto' is what produced the reported scores",
    )
    predict.add_argument(
        "--expansion-iterations", type=int, default=20,
        help="rounds of set expansion for awardWonBy under --proposer auto",
    )
    predict.add_argument("--relations", nargs="*", default=None, choices=list(RELATIONS))
    predict.add_argument("--limit", type=int, default=0, help="rows per relation, 0 = all")
    predict.add_argument("--samples", type=int, default=0, help="0 = per-relation default")
    predict.add_argument("--temperature", type=float, default=0.8)
    predict.add_argument("--shots", type=int, default=6)
    predict.add_argument("--seed", type=int, default=0)
    predict.add_argument("--concurrency", type=int, default=8, help="in-flight requests")
    predict.add_argument("--row-concurrency", type=int, default=12)
    predict.add_argument("--calibrator", default=str(RUNS_DIR / "calibrator.json"))
    predict.add_argument(
        "--proposer",
        default="sampling",
        choices=["sampling", "mcts", "expansion", "mcts+expansion", "auto", "structured"],
        help="stage-1 mechanism; stages 2-4 are identical across all of them",
    )
    predict.add_argument("--mcts-iterations", type=int, default=20)
    predict.add_argument(
        "--missing-mass-cap",
        type=float,
        default=None,
        help="override the swept value; 0 disables the missing-mass prior",
    )
    predict.add_argument("--no-verify", action="store_true")
    predict.add_argument("--no-count", action="store_true")
    predict.add_argument("--no-cache", action="store_true")
    predict.add_argument("--fresh", action="store_true", help="ignore the checkpoint")
    predict.add_argument("--allow-over-budget", action="store_true")
    predict.add_argument("--tag", default=None)
    predict.add_argument("--out", default=None)

    fit = sub.add_parser("fit", help="fit the calibrator from a labelled checkpoint")
    fit.add_argument("--split", default="train", choices=["train", "val"])
    fit.add_argument("--checkpoint", required=True)
    fit.add_argument("--l2", type=float, default=1.0)
    fit.add_argument("--out", default=str(RUNS_DIR / "calibrator.json"))

    score = sub.add_parser("score", help="score a prediction file")
    score.add_argument("--split", default="val", choices=["train", "val"])
    score.add_argument("--predictions", required=True)
    score.add_argument("--worst", type=int, default=0)

    sweep_cmd = sub.add_parser("sweep", help="grid-search decoder settings on a split")
    sweep_cmd.add_argument("--split", default="train", choices=["train", "val"])
    sweep_cmd.add_argument("--checkpoint", required=True)
    sweep_cmd.add_argument("--top", type=int, default=12)
    sweep_cmd.add_argument("--out", default=str(RUNS_DIR / "calibrator.json"))

    sub.add_parser("models", help="list open-weight models and budgets")

    validate = sub.add_parser("validate", help="check a submission file before sending")
    validate.add_argument("--split", default="test", choices=["train", "val", "test"])
    validate.add_argument("--predictions", required=True)

    package = sub.add_parser(
        "package", help="validate and zip a prediction file for Codabench"
    )
    package.add_argument("--split", default="test", choices=["train", "val", "test"])
    package.add_argument("--predictions", required=True)
    package.add_argument("--out", default=None)

    ablate = sub.add_parser("ablate", help="re-decode a checkpoint under several rules")
    ablate.add_argument("--split", default="val", choices=["train", "val"])
    ablate.add_argument("--checkpoint", required=True)
    ablate.add_argument("--calibrator", default=str(RUNS_DIR / "calibrator.json"))

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if args.command == "predict":
        return asyncio.run(cmd_predict(args))
    if args.command == "fit":
        return cmd_fit(args)
    if args.command == "score":
        return cmd_score(args)
    if args.command == "ablate":
        return cmd_ablate(args)
    if args.command == "sweep":
        return cmd_sweep(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "package":
        return cmd_package(args)
    if args.command == "models":
        from .clients import describe_choices

        print(describe_choices())
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
