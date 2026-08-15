# Normalisation and grouping of candidates into strings and numeric clusters.
from __future__ import annotations

import importlib.util
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

_EVALUATE_PATH = Path(__file__).resolve().parent.parent / "dataset2026-main" / "evaluate.py"


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("lmkbc_evaluate", _EVALUATE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load evaluator from {_EVALUATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("lmkbc_evaluate", module)
    spec.loader.exec_module(module)
    return module


_EVAL = _load_evaluator()
normalize_string = _EVAL.normalize_string
string_true_positives = _EVAL.string_true_positives
numeric_true_positives = _EVAL.numeric_true_positives

NUMERIC_TOLERANCE = 0.05

_NUMBER_RE = re.compile(r"-?\d[\d,\s]*(?:\.\d+)?")
_SCALE_WORDS = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "m": 1e6,
    "mn": 1e6,
    "billion": 1e9,
    "bn": 1e9,
}
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)\s*$")


@dataclass
class Candidate:

    label: str
    variants: list[str] = field(default_factory=list)
    support: int = 0
    n_samples: int = 1
    value: float | None = None
    prob: float = 0.0
    features: dict[str, float] = field(default_factory=dict)

    @property
    def frequency(self) -> float:
        return self.support / max(self.n_samples, 1)

    @property
    def smoothed_frequency(self) -> float:
        return (self.support + 0.5) / (self.n_samples + 1.0)


def parse_number(text: str) -> float | None:
    if text is None:
        return None
    raw = str(text).strip().lower().replace("−", "-")
    if not raw:
        return None

    match = _NUMBER_RE.search(raw)
    if not match:
        return None
    try:
        value = float(re.sub(r"[,\s]", "", match.group(0)))
    except ValueError:
        return None

    tail = raw[match.end() :].strip()
    for word, scale in _SCALE_WORDS.items():
        if tail.startswith(word) and (len(tail) == len(word) or not tail[len(word)].isalnum()):
            value *= scale
            break
    return value


def canonical_string(text: str) -> str:
    out = str(text or "").strip()
    out = _PARENTHETICAL_RE.sub("", out)
    out = out.strip(" \t\n.,;:")
    return " ".join(out.split())


def cluster_strings(samples: Sequence[Sequence[str]]) -> list[Candidate]:
    n_samples = max(len(samples), 1)
    support: dict[str, int] = defaultdict(int)
    surface: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for answers in samples:
        seen_in_sample: set[str] = set()
        for answer in answers:
            label = canonical_string(answer)
            if not label:
                continue
            key = normalize_string(label)
            if not key:
                continue
            surface[key][label] += 1
            if key not in seen_in_sample:
                seen_in_sample.add(key)
                support[key] += 1

    candidates = []
    for key, count in support.items():
        forms = sorted(surface[key].items(), key=lambda kv: (-kv[1], len(kv[0])))
        candidates.append(
            Candidate(
                label=forms[0][0],
                variants=[f for f, _ in forms],
                support=count,
                n_samples=n_samples,
            )
        )
    return sorted(candidates, key=lambda c: -c.support)


def cluster_numbers(
    samples: Sequence[Sequence[str]], tolerance: float = NUMERIC_TOLERANCE
) -> list[Candidate]:
    n_samples = max(len(samples), 1)
    values: list[float] = []
    for answers in samples:
        best: float | None = None
        for answer in answers:
            parsed = parse_number(answer)
            if parsed is not None and parsed > 0:
                best = parsed
                break
        if best is not None:
            values.append(best)

    if not values:
        return []

    width = math.log1p(tolerance)
    ordered = sorted(values)
    clusters: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if math.log(value) - math.log(clusters[-1][-1]) <= width:
            clusters[-1].append(value)
        else:
            clusters.append([value])

    candidates = []
    for group in clusters:
        median = sorted(group)[len(group) // 2]
        label = format_number(median)
        candidates.append(
            Candidate(
                label=label,
                variants=[format_number(v) for v in group],
                support=len(group),
                n_samples=n_samples,
                value=median,
            )
        )
    return sorted(candidates, key=lambda c: -c.support)


def format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


def within_tolerance(a: float, b: float, tolerance: float = NUMERIC_TOLERANCE) -> bool:
    return b != 0 and abs(a - b) / abs(b) <= tolerance


def score_candidates_against_gold(
    candidates: Sequence[Candidate],
    gold: Sequence[Sequence[str]],
    *,
    numeric: bool,
) -> list[int]:
    if numeric:
        gold_values = [parse_number(aliases[0]) for aliases in gold if aliases]
        labels = []
        for candidate in candidates:
            value = candidate.value if candidate.value is not None else parse_number(candidate.label)
            hit = value is not None and any(
                g is not None and within_tolerance(value, g) for g in gold_values
            )
            labels.append(int(hit))
        return labels

    gold_keys = [{normalize_string(a) for a in aliases} for aliases in gold]
    labels = []
    for candidate in candidates:
        key = normalize_string(candidate.label)
        labels.append(int(any(key in keys for keys in gold_keys)))
    return labels


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = normalize_string(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out
