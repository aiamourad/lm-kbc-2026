#!/usr/bin/env python3
# Runs the final system end to end for one split.
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

BUDGETS = [
    (["hasArea"], 64, None, 1.3),
    (["hasCapacity", "awardWonBy"], 64, None, None),
    (["countryLandBordersCountry", "personHasCityOfDeath"], 128, 12, None),
    (["companyTradesAtStockExchange"], 256, 12, None),
]

SUPPORT_FLAG = {
    "awardWonBy": "--award-support",
    "companyTradesAtStockExchange": "--company-support",
    "countryLandBordersCountry": "--border-support",
    "personHasCityOfDeath": "--death-support",
}


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Runs the final system end to end for one split.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data", default=str(HERE.parent / "dataset2026-main" / "data"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workdir", default="out")
    ap.add_argument("--supports", default=str(HERE / "fitted-supports.json"))
    ap.add_argument("--fit", action="store_true",
                    help="refit the thresholds on train first, for a checkpoint "
                         "other than the one they were fitted on")
    ap.add_argument("--skip-stages", action="store_true",
                    help="core pipeline only, without the rescue and the gate")
    args = ap.parse_args()

    data = Path(args.data)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(args.model).name}-{args.split}"
    out = Path(args.out) if args.out else work / f"{tag}.jsonl"
    py = sys.executable

    if args.fit:
        dumps = []
        for relations, samples, shots, _temperature in BUDGETS:
            sets = [r for r in relations if r in SUPPORT_FLAG]
            if not sets:
                continue
            dump = work / f"{tag}-fit-{samples}.samples.jsonl"
            cmd = [py, str(HERE / "completion_system.py"),
                   "--train", str(data / "train.jsonl"), "--input", str(data / "train.jsonl"),
                   "--output", str(work / f"{tag}-fit-{samples}.jsonl"),
                   "--dump-samples", str(dump), "--relations", ",".join(sets),
                   "--base-url", args.base_url, "--model", args.model,
                   "--samples", str(samples)]
            if shots:
                cmd += ["--set-shots", str(shots)]
            run(cmd)
            dumps.append(dump)
        merged = work / f"{tag}-fit.samples.jsonl"
        merged.write_text("".join(d.read_text(encoding="utf-8") for d in dumps), encoding="utf-8")
        args.supports = str(work / f"{tag}-supports.json")
        run([py, str(HERE / "aggregation.py"), "-g", str(data / "train.jsonl"),
             "-s", str(merged), "--out", args.supports])

    supports = json.loads(Path(args.supports).read_text(encoding="utf-8"))

    predictions: list[dict] = []
    sample_dumps: list[Path] = []
    for relations, samples, shots, temperature in BUDGETS:
        stem = f"{tag}-n{samples}" + (f"-t{temperature}" if temperature else "")
        piece = work / f"{stem}.jsonl"
        dump = work / f"{stem}.samples.jsonl"
        cmd = [py, str(HERE / "completion_system.py"),
               "--train", str(data / "train.jsonl"), "--input", str(data / f"{args.split}.jsonl"),
               "--output", str(piece), "--dump-samples", str(dump),
               "--relations", ",".join(relations),
               "--base-url", args.base_url, "--model", args.model,
               "--samples", str(samples)]
        if shots:
            cmd += ["--set-shots", str(shots)]
        if temperature:
            cmd += ["--temperature", str(temperature)]
        for relation in relations:
            if relation in SUPPORT_FLAG and relation in supports:
                cmd += [SUPPORT_FLAG[relation], str(supports[relation])]
        run(cmd)
        predictions += read_jsonl(piece)
        sample_dumps.append(dump)

    core = work / f"{tag}.core.jsonl"
    with open(core, "w", encoding="utf-8") as fh:
        for row in predictions:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"core pipeline: {len(predictions)} rows -> {core}", file=sys.stderr)

    if args.skip_stages:
        out.write_text(core.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        award_dump = work / f"{tag}-n64.samples.jsonl"
        ver = work / f"{tag}.ver.json"
        alive = work / f"{tag}.alive.json"
        run([py, str(HERE / "postprocess.py"), "rescue", "--dump", str(award_dump),
             "--relation", "awardWonBy", "--gold", str(data / f"{args.split}.jsonl"),
             "--train", str(data / "train.jsonl"), "--base-url", args.base_url,
             "--model", args.model, "--samples", "8", "--out", str(ver)])
        run([py, str(HERE / "postprocess.py"), "existence-check", "--gold", str(data / f"{args.split}.jsonl"),
             "--train", str(data / "train.jsonl"), "--base-url", args.base_url,
             "--model", args.model, "--samples", "16", "--out", str(alive)])
        run([py, str(HERE / "postprocess.py"), "apply", "--pred", str(core),
             "--ver", str(ver), "--existence", str(alive),
             "--train", str(data / "train.jsonl"),
             "--samples", "64", "--out", str(out)])

    print(f"\npredictions: {out}", file=sys.stderr)
    if args.split != "test":
        run([py, str(HERE / "scorer.py"), "-g", str(data / f"{args.split}.jsonl"),
             "-p", str(out)])
    else:
        print("test gold is not public; package the file and submit it:\n"
              f"  python ../scripts/package_submission.py -p {out} "
              f"-g {data / 'test.jsonl'} -o submission.zip", file=sys.stderr)


if __name__ == "__main__":
    main()
