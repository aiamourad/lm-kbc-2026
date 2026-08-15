# Verification, per-relation calibration, and expected-F1 decoding.
from __future__ import annotations

import json
import math
from dataclasses import field
from pathlib import Path
from typing import Literal, Mapping, Sequence

from .data import RELATION_TYPE, Row
from .group import (
    dedupe_preserving_order,
    numeric_true_positives,
    string_true_positives,
)


class RowScore:

    subject: str
    relation: str
    precision: float
    recall: float
    f1: float
    tp: int
    n_pred: int
    n_gold: int

    @property
    def gold_empty(self) -> bool:
        return self.n_gold == 0

    @property
    def pred_empty(self) -> bool:
        return self.n_pred == 0

def score_row(row: Row, predictions: Sequence[str]) -> RowScore:
    preds = dedupe_preserving_order(str(p) for p in predictions if isinstance(p, str))
    gold = [list(aliases) for aliases in row.gold]
    numeric = RELATION_TYPE.get(row.relation, "string") == "numeric"

    if not gold:
        tp = 0
    elif numeric:
        tp = numeric_true_positives(preds, gold, 0.05)
    else:
        tp = string_true_positives(preds, gold)

    precision = tp / len(preds) if preds else 1.0
    recall = tp / len(gold) if gold else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return RowScore(
        subject=row.subject,
        relation=row.relation,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        n_pred=len(preds),
        n_gold=len(gold),
    )

class RelationReport:

    relation: str
    n: int
    macro_p: float
    macro_r: float
    macro_f1: float
    avg_preds: float
    n_empty_pred: int
    n_empty_gold: int
    empty_correct: int

    @property
    def abstention_recall(self) -> float:
        return self.empty_correct / self.n_empty_gold if self.n_empty_gold else 0.0

def score_all(
    rows: Sequence[Row], predictions: Mapping[tuple[str, str], Sequence[str]]
) -> tuple[list[RowScore], dict[str, RelationReport], RelationReport]:
    scores = [score_row(row, predictions.get(row.key, [])) for row in rows]

    by_relation: dict[str, RelationReport] = {}
    for relation in sorted({s.relation for s in scores}):
        subset = [s for s in scores if s.relation == relation]
        by_relation[relation] = _summarise(relation, subset)

    return scores, by_relation, _summarise("*** All Relations ***", scores)

def _summarise(name: str, scores: Sequence[RowScore]) -> RelationReport:
    n = len(scores)
    empty_gold = [s for s in scores if s.gold_empty]
    return RelationReport(
        relation=name,
        n=n,
        macro_p=sum(s.precision for s in scores) / n if n else 0.0,
        macro_r=sum(s.recall for s in scores) / n if n else 0.0,
        macro_f1=sum(s.f1 for s in scores) / n if n else 0.0,
        avg_preds=sum(s.n_pred for s in scores) / n if n else 0.0,
        n_empty_pred=sum(1 for s in scores if s.pred_empty),
        n_empty_gold=len(empty_gold),
        empty_correct=sum(1 for s in empty_gold if s.pred_empty),
    )

def format_report(
    by_relation: Mapping[str, RelationReport], overall: RelationReport
) -> str:
    header = (
        f"{'relation':32s} {'macro-p':>8s} {'macro-r':>8s} {'macro-f1':>9s} "
        f"{'#preds':>7s} {'empty-pred':>11s} {'abstain-rec':>12s}"
    )
    lines = [header, "-" * len(header)]
    for report in list(by_relation.values()) + [overall]:
        abstain = (
            f"{report.empty_correct}/{report.n_empty_gold}"
            if report.n_empty_gold
            else "-"
        )
        lines.append(
            f"{report.relation:32s} {report.macro_p:8.3f} {report.macro_r:8.3f} "
            f"{report.macro_f1:9.3f} {report.avg_preds:7.2f} {report.n_empty_pred:11d} "
            f"{abstain:>12s}"
        )
    return "\n".join(lines)

def oracle_scores(rows: Sequence[Row], candidate_pools: Mapping[tuple[str, str], Sequence[str]]) -> dict[str, float]:
    per_relation: dict[str, list[float]] = {}
    for row in rows:
        pool = list(candidate_pools.get(row.key, []))
        best = 0.0
        for k in range(len(pool) + 1):
            best = max(best, score_row(row, pool[:k]).f1)
        per_relation.setdefault(row.relation, []).append(best)
    return {
        relation: sum(values) / len(values) for relation, values in per_relation.items()
    }


FEATURES = ("recurrence", "verification", "rank_fraction", "pool_fraction")

DEFAULT_WEIGHTS = (0.0, 0.5, 0.5, 0.0, 0.0)

def _design(features: dict[str, float]) -> list[float]:
    row = [1.0]
    for name in FEATURES:
        row.append(logit(float(features.get(name, 0.5))))
    return row

class Calibrator:

    weights: dict[str, list[float]] = field(default_factory=dict)
    fallback: list[float] = field(default_factory=lambda: list(DEFAULT_WEIGHTS))
    n_fit: dict[str, int] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Calibrator":
        return cls()

    def probability(self, relation: str, features: dict[str, float]) -> float:
        weights = self.weights.get(relation, self.fallback)
        vector = _design(features)
        z = sum(w * x for w, x in zip(weights, vector))
        return min(max(sigmoid(z), 1e-4), 1 - 1e-4)

    def fit_relation(
        self,
        relation: str,
        features: Sequence[dict[str, float]],
        labels: Sequence[int],
        *,
        l2: float = 1.0,
        iterations: int = 3000,
        learning_rate: float = 0.1,
    ) -> None:
        if not features:
            return
        design = [_design(f) for f in features]
        targets = [float(y) for y in labels]
        n_features = len(design[0])
        weights = [0.0] * n_features
        n = len(design)

        for _ in range(iterations):
            gradient = [0.0] * n_features
            for vector, target in zip(design, targets):
                z = sum(w * x for w, x in zip(weights, vector))
                error = sigmoid(z) - target
                for j, x in enumerate(vector):
                    gradient[j] += error * x
            for j in range(n_features):
                penalty = l2 * weights[j] if j > 0 else 0.0
                weights[j] -= learning_rate * (gradient[j] + penalty) / n

        self.weights[relation] = weights
        self.n_fit[relation] = n

    def report(self) -> str:
        names = "".join(f"{('w_' + f)[:10]:>11s}" for f in FEATURES)
        header = f"{'relation':32s} {'n':>5s} {'intercept':>10s}{names}"
        lines = [header, "-" * len(header)]
        for relation, weights in sorted(self.weights.items()):
            n = self.n_fit.get(relation, 0)
            cells = "".join(f"{w:+11.3f}" for w in weights[1:])
            lines.append(f"{relation:32s} {n:5d} {weights[0]:+10.3f}{cells}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"weights": self.weights, "fallback": self.fallback, "n_fit": self.n_fit},
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            weights={k: list(v) for k, v in data.get("weights", {}).items()},
            fallback=list(data.get("fallback", DEFAULT_WEIGHTS)),
            n_fit=data.get("n_fit", {}),
        )

def reliability(
    probabilities: Sequence[float], labels: Sequence[int], *, bins: int = 10
) -> str:
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, label in zip(probabilities, labels):
        index = min(int(probability * bins), bins - 1)
        buckets[index].append((probability, label))

    lines = ["  bin        n    mean_p   observed    gap"]
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_p = sum(p for p, _ in bucket) / len(bucket)
        observed = sum(y for _, y in bucket) / len(bucket)
        lines.append(
            f"  {index/bins:.1f}-{(index+1)/bins:.1f} {len(bucket):6d}   "
            f"{mean_p:6.3f}    {observed:6.3f}  {observed-mean_p:+6.3f}"
        )
    total = len(probabilities)
    if total:
        brier = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / total
        logloss = -sum(
            y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
            for p, y in zip(probabilities, labels)
        ) / total
        lines.append(f"  brier={brier:.4f}  logloss={logloss:.4f}  n={total}")
    return "\n".join(lines)


INDEPENDENT = "independent"

EXCLUSIVE = "exclusive"

Mode = Literal["independent", "exclusive"]

def poisson_binomial(probs: Sequence[float]) -> list[float]:
    pmf = [1.0]
    for p in probs:
        p = min(max(float(p), 0.0), 1.0)
        nxt = [0.0] * (len(pmf) + 1)
        for j, mass in enumerate(pmf):
            if mass == 0.0:
                continue
            nxt[j] += mass * (1.0 - p)
            nxt[j + 1] += mass * p
        pmf = nxt
    return pmf

def _convolve(a: Sequence[float], b: Sequence[float]) -> list[float]:
    if len(b) == 1:
        return [x * b[0] for x in a]
    out = [0.0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0.0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out

def _truncate(pmf: Sequence[float], epsilon: float = 1e-9) -> list[float]:
    out = list(pmf)
    while len(out) > 1 and out[-1] < epsilon:
        out.pop()
    total = sum(out)
    return [x / total for x in out] if total > 0 else out

class DecodeResult:

    selected: list[str]
    expected_f1: float
    k: int
    curve: list[float] = field(default_factory=list)
    margin: float = 0.0

def expected_f1_curve(
    probs: Sequence[float],
    *,
    mode: Mode = INDEPENDENT,
    missing_mass: Sequence[float] | None = None,
    gold_is_nonempty: bool = False,
) -> list[float]:
    probs = [min(max(float(p), 0.0), 1.0) for p in probs]
    n = len(probs)

    if mode == EXCLUSIVE:
        total = sum(probs)
        if total > 1.0:
            probs = [p / total for p in probs]
            total = 1.0
        p_empty = 0.0 if gold_is_nonempty else max(0.0, 1.0 - total)

        curve = [p_empty]
        cumulative = 0.0
        for k in range(1, n + 1):
            cumulative += probs[k - 1]
            curve.append(cumulative * 2.0 / (k + 1))
        return curve

    miss = _truncate(missing_mass) if missing_mass else [1.0]

    suffix: list[list[float]] = [[1.0]] * (n + 1)
    running = [1.0]
    for i in range(n - 1, -1, -1):
        running = _truncate(_convolve(running, [1.0 - probs[i], probs[i]]))
        suffix[i] = running

    curve: list[float] = []
    prefix = [1.0]
    for k in range(n + 1):
        if k > 0:
            prefix = _truncate(_convolve(prefix, [1.0 - probs[k - 1], probs[k - 1]]))
        outside = _convolve(suffix[k], miss)

        if k == 0:
            curve.append(0.0 if gold_is_nonempty else outside[0])
            continue

        total = 0.0
        for t, p_t in enumerate(prefix):
            if p_t == 0.0 or t == 0:
                continue
            for w, p_w in enumerate(outside):
                if p_w == 0.0:
                    continue
                total += p_t * p_w * (2.0 * t) / (k + t + w)
        curve.append(total)
    return curve

def decode(
    labels: Sequence[str],
    probs: Sequence[float],
    *,
    mode: Mode = INDEPENDENT,
    missing_mass: Sequence[float] | None = None,
    gold_is_nonempty: bool = False,
    max_k: int | None = None,
) -> DecodeResult:
    if len(labels) != len(probs):
        raise ValueError(f"{len(labels)} labels but {len(probs)} probabilities")
    if not labels:
        return DecodeResult(selected=[], expected_f1=0.0 if gold_is_nonempty else 1.0, k=0)

    order = sorted(range(len(labels)), key=lambda i: -probs[i])
    sorted_labels = [labels[i] for i in order]
    sorted_probs = [probs[i] for i in order]

    curve = expected_f1_curve(
        sorted_probs,
        mode=mode,
        missing_mass=missing_mass,
        gold_is_nonempty=gold_is_nonempty,
    )

    limit = len(sorted_labels) if max_k is None else min(max_k, len(sorted_labels))
    feasible = curve[: limit + 1]
    best_k = max(range(len(feasible)), key=lambda k: feasible[k])

    runner_up = sorted((v for k, v in enumerate(feasible) if k != best_k), reverse=True)
    margin = feasible[best_k] - (runner_up[0] if runner_up else 0.0)

    return DecodeResult(
        selected=sorted_labels[:best_k],
        expected_f1=feasible[best_k],
        k=best_k,
        curve=curve,
        margin=margin,
    )

def geometric_missing_mass(expected_missing: float, cap: int = 60) -> list[float]:
    if expected_missing <= 0:
        return [1.0]
    ratio = expected_missing / (1.0 + expected_missing)
    pmf = [(1.0 - ratio) * ratio**m for m in range(cap + 1)]
    total = sum(pmf)
    return [x / total for x in pmf]

def f1(selected: Sequence[str], n_gold: int, n_matched: int) -> float:
    k = len(selected)
    if k == 0 and n_gold == 0:
        return 1.0
    if k == 0 or n_gold == 0:
        return 0.0
    precision = n_matched / k
    recall = n_matched / n_gold
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
