# Calibrate: fits the per-relation logistic regression offline from one traced run.
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .decode import Calibrator
from .data import RELATIONS, Row, load_split
from .decode import EXCLUSIVE, INDEPENDENT, decode, geometric_missing_mass
from .group import Candidate, score_candidates_against_gold
from .decode import score_all
from .propose import _EXCLUSIVE_RELATIONS

POOLED = "__pooled__"


def load_checkpoint(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def rebuild_candidates(record: dict[str, Any]) -> list[Candidate]:
    return [
        Candidate(
            label=c["label"],
            variants=list(c.get("variants") or []),
            support=c["support"],
            n_samples=c["n_samples"],
            value=c.get("value"),
            features=dict(c.get("features") or {}),
        )
        for c in record.get("candidates") or []
    ]


def candidate_features(
    candidate: Candidate, rank: int = 0, pool_size: int = 1
) -> dict[str, float]:
    features = dict(candidate.features)
    features["recurrence"] = candidate.smoothed_frequency
    features.setdefault("verification", 0.5)
    features["rank_fraction"] = 1.0 / (1.0 + rank)
    features["pool_fraction"] = 1.0 / (1.0 + max(pool_size, 1))
    features.setdefault("gate", 0.5)
    return features


def fit_calibrator(
    records: Sequence[dict[str, Any]],
    rows: dict[tuple[str, str], Row],
    *,
    l2: float = 1.0,
    calibration_floor: int = 100,
) -> Calibrator:
    by_relation: dict[str, tuple[list[dict[str, float]], list[int]]] = {}
    for record in records:
        row = rows.get((record["subject"], record["relation"]))
        if row is None:
            continue
        candidates = rebuild_candidates(record)
        if not candidates:
            continue
        hits = score_candidates_against_gold(
            candidates, row.gold, numeric=RELATIONS[record["relation"]].is_numeric
        )
        features, labels = by_relation.setdefault(record["relation"], ([], []))
        for rank, (candidate, hit) in enumerate(zip(candidates, hits)):
            features.append(candidate_features(candidate, rank, len(candidates)))
            labels.append(hit)

    calibrator = Calibrator()

    pooled_features: list[dict[str, float]] = []
    pooled_labels: list[int] = []
    for features, labels in by_relation.values():
        pooled_features.extend(features)
        pooled_labels.extend(labels)
    if pooled_features:
        calibrator.fit_relation(POOLED, pooled_features, pooled_labels, l2=l2)
        calibrator.fallback = list(calibrator.weights[POOLED])

    for relation, (features, labels) in by_relation.items():
        if len(features) >= calibration_floor and 0 < sum(labels) < len(labels):
            calibrator.fit_relation(relation, features, labels, l2=l2)
    return calibrator


def decode_records(
    records: Sequence[dict[str, Any]],
    calibrator: Calibrator,
    *,
    missing_mass_cap: float = 1.0,
    use_verify: bool = True,
) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for record in records:
        relation = record["relation"]
        spec = RELATIONS[relation]
        candidates = rebuild_candidates(record)
        key = (record["subject"], relation)
        if not candidates:
            out[key] = []
            continue

        for rank, candidate in enumerate(candidates):
            features = candidate_features(candidate, rank, len(candidates))
            if not use_verify:
                features["verification"] = 0.5
            candidate.prob = calibrator.probability(relation, features)

        mode = EXCLUSIVE if relation in _EXCLUSIVE_RELATIONS else INDEPENDENT
        missing = None
        count_estimate = record.get("count_estimate")
        if mode == INDEPENDENT and count_estimate and missing_mass_cap > 0:
            present = sum(c.prob for c in candidates)
            shortfall = max(0.0, float(count_estimate) - present)
            missing = geometric_missing_mass(min(shortfall, missing_mass_cap * present))

        out[key] = decode(
            [c.label for c in candidates],
            [c.prob for c in candidates],
            mode=mode,
            missing_mass=missing,
            gold_is_nonempty=not spec.allows_empty,
            max_k=spec.max_candidates,
        ).selected
    return out


@dataclass
class SweepPoint:

    l2: float
    calibration_floor: int
    missing_mass_cap: float
    use_verify: bool
    macro_f1: float
    per_relation: dict[str, float]

    def label(self) -> str:
        return (
            f"l2={self.l2:<4g} floor={self.calibration_floor:<4d} "
            f"cap={self.missing_mass_cap:<4g} verify={'on ' if self.use_verify else 'off'}"
        )


def sweep(
    checkpoint: str | Path,
    split: str = "train",
    *,
    l2_values: Iterable[float] = (0.1, 1.0),
    floors: Iterable[int] = (0, 100, 100000),
    caps: Iterable[float] = (0.0, 0.25, 0.5, 1.0),
    verify_values: Iterable[bool] = (True, False),
) -> list[SweepPoint]:
    rows = {r.key: r for r in load_split(split)}
    records = [r for r in load_checkpoint(checkpoint) if (r["subject"], r["relation"]) in rows]
    covered = [rows[(r["subject"], r["relation"])] for r in records]

    results: list[SweepPoint] = []
    for l2 in l2_values:
        for floor in floors:
            calibrator = fit_calibrator(records, rows, l2=l2, calibration_floor=floor)
            for cap in caps:
                for use_verify in verify_values:
                    predictions = decode_records(
                        records, calibrator, missing_mass_cap=cap, use_verify=use_verify
                    )
                    _, by_relation, overall = score_all(covered, predictions)
                    results.append(
                        SweepPoint(
                            l2=l2,
                            calibration_floor=floor,
                            missing_mass_cap=cap,
                            use_verify=use_verify,
                            macro_f1=overall.macro_f1,
                            per_relation={
                                name: report.macro_f1
                                for name, report in by_relation.items()
                            },
                        )
                    )
    results.sort(key=lambda p: -p.macro_f1)
    return results


def format_sweep(results: Sequence[SweepPoint], top: int = 12) -> str:
    relations = sorted(results[0].per_relation) if results else []
    header = f"{'configuration':52s} {'macro-f1':>9s}" + "".join(
        f"{name[:11]:>13s}" for name in relations
    )
    lines = [header, "-" * len(header)]
    for point in results[:top]:
        cells = "".join(f"{point.per_relation.get(n, 0.0):13.3f}" for n in relations)
        lines.append(f"{point.label():52s} {point.macro_f1:9.3f}{cells}")
    return "\n".join(lines)
