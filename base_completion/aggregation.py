# Tolerance clustering, set recurrence, and fitting the per-relation threshold on train.
from __future__ import annotations

import re
from dataclasses import dataclass
import json
import unicodedata
from collections import Counter
from typing import Sequence
import argparse
import sys
from pathlib import Path
from scorer import load_official


SQ_MILE_IN_SQ_KM = 2.589988110336
HECTARE_IN_SQ_KM = 0.01
ACRE_IN_SQ_KM = 0.0040468564224

TOLERANCE = 0.05


@dataclass(frozen=True)
class QuestionForm:
    name: str
    system: str
    template: str

    scale: float = 1.0

    freeform: bool = False

    parse_units: bool = False

    def render(self, subject: str) -> str:
        return self.template.format(subject=subject)


_JSON_TAIL = (
    '\n\nReturn JSON: {{"value": <number>}} with a bare number and no units, '
    "no commas, no range."
)


NUMERIC_SYSTEM = (
    "You are a precise reference work. You report published figures from "
    "memory. When you are unsure, you still give your single best numeric "
    "estimate rather than refusing. You never return a range."
)


AREA_FORMS: tuple[QuestionForm, ...] = (
    QuestionForm(
        name="direct",
        system=NUMERIC_SYSTEM,
        template=(
            "What is the total surface area of {subject}, in square kilometres?"
            "\n\nFor a lake use the water surface area, not the drainage basin. "
            "For an island use the island's own area, not its municipality or "
            "island group. For a country use total area including inland water."
            + _JSON_TAIL
        ),
    ),
    QuestionForm(
        name="infobox",
        system=NUMERIC_SYSTEM,
        template=(
            "Recall the encyclopedia article for {subject}. The infobox has an "
            "'Area' field.\n\nReproduce the number in that field, converted to "
            "square kilometres. Report the published figure as written, "
            "including its decimal places, rather than a rounded estimate."
            + _JSON_TAIL
        ),
    ),
    QuestionForm(
        name="imperial",
        system=NUMERIC_SYSTEM,
        scale=SQ_MILE_IN_SQ_KM,
        template=(
            "What is the total surface area of {subject}, in SQUARE MILES?"
            "\n\nAnswer in square miles, not square kilometres."
            + _JSON_TAIL
        ),
    ),
    QuestionForm(
        name="anchor",
        system=NUMERIC_SYSTEM,
        template=(
            "Consider {subject}.\n\n"
            "1. What kind of geographic entity is it, and where is it?\n"
            "2. Name a well-known entity of the same kind and clearly similar "
            "size, and state that one's area in square kilometres.\n"
            "3. Is {subject} larger or smaller than it, and by roughly what "
            "factor?\n"
            "4. Give the area of {subject} in square kilometres.\n\n"
            'Return JSON: {{"value": <number>}} holding only the answer to '
            "step 4."
        ),
    ),
    QuestionForm(
        name="freeform",
        system=NUMERIC_SYSTEM,
        freeform=True,
        template=(
            "Complete this sentence with the published figure.\n\n"
            "The total area of {subject} is"
        ),
    ),
    QuestionForm(
        name="hectare",
        system=NUMERIC_SYSTEM,
        scale=HECTARE_IN_SQ_KM,
        template=(
            "What is the total area of {subject}, in HECTARES?\n\n"
            "Answer in hectares, not square kilometres." + _JSON_TAIL
        ),
    ),
    QuestionForm(
        name="state",
        system=NUMERIC_SYSTEM,
        parse_units=True,
        template=(
            "State the published total area of {subject} exactly as a "
            "reference work gives it: one number followed by its unit "
            "(square kilometres, square miles, hectares or acres). No other "
            "words."
        ),
    ),
)


CAPACITY_SYSTEM = (
    "You are a precise sports and venue reference. You report published "
    "seating capacities from memory. When unsure you still give your single "
    "best numeric estimate rather than refusing. You never return a range."
)


CAPACITY_FORMS: tuple[QuestionForm, ...] = (
    QuestionForm(
        name="direct",
        system=CAPACITY_SYSTEM,
        template=(
            "What is the spectator capacity of {subject}?\n\n"
            "Use the published venue capacity, not record attendance, not a "
            "concert configuration, not the city population."
            + _JSON_TAIL
        ),
    ),
    QuestionForm(
        name="infobox",
        system=CAPACITY_SYSTEM,
        template=(
            "Recall the encyclopedia article for {subject}. The infobox has a "
            "'Capacity' field.\n\nReproduce that number exactly as published, "
            "including its exact digits rather than a round approximation."
            + _JSON_TAIL
        ),
    ),
    QuestionForm(
        name="decompose",
        system=CAPACITY_SYSTEM,
        template=(
            "Consider {subject}.\n\n"
            "1. What sport or use is the venue built for, and who is the home "
            "team or main occupant?\n"
            "2. What competition tier does that occupant play at?\n"
            "3. What is the typical capacity range for a venue at that tier in "
            "that country?\n"
            "4. What is this specific venue's published capacity?\n\n"
            'Return JSON: {{"value": <number>}} holding only the answer to '
            "step 4."
        ),
    ),
    QuestionForm(
        name="anchor",
        system=CAPACITY_SYSTEM,
        template=(
            "Consider {subject}.\n\n"
            "1. Name another venue in the same city or region whose capacity "
            "you are confident of, and give it.\n"
            "2. Is {subject} bigger or smaller, and roughly by how much?\n"
            "3. Give the capacity of {subject}.\n\n"
            'Return JSON: {{"value": <number>}} holding only the answer to '
            "step 3."
        ),
    ),
    QuestionForm(
        name="freeform",
        system=CAPACITY_SYSTEM,
        freeform=True,
        template=(
            "Complete this sentence with the published figure.\n\n"
            "The seating capacity of {subject} is"
        ),
    ),
    QuestionForm(
        name="state",
        system=CAPACITY_SYSTEM,
        parse_units=True,
        template=(
            "State the published spectator capacity of {subject} exactly as a "
            "reference work gives it: one number, nothing else."
        ),
    ),
)


_NUMBER = re.compile(
    r"[-+]?\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?"
)

_UNIT_SCALES: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\b(?:sq\.?\s*mi|square\s+miles?|mi²|mi2)\b", re.I), SQ_MILE_IN_SQ_KM),
    (re.compile(r"\b(?:hectares?|ha)\b", re.I), HECTARE_IN_SQ_KM),
    (re.compile(r"\b(?:acres?)\b", re.I), ACRE_IN_SQ_KM),
)


def parse_number(text: str, *, detect_units: bool = False) -> float | None:
    text = str(text or "")
    match = _NUMBER.search(text)

    if not match:
        return None

    try:
        value = float(re.sub(r"[,\s]", "", match.group(0)))
    except ValueError:
        return None

    if detect_units:
        tail = text[match.end() : match.end() + 40]
        next_digit = re.search(r"\d", tail)

        if next_digit:
            tail = tail[: next_digit.start()]

        for pattern, scale in _UNIT_SCALES:
            if pattern.search(tail):
                return value * scale

    return value


def cluster_vote(
    values: list[float],
    *,
    tolerance: float = TOLERANCE,
) -> tuple[float | None, float]:
    usable = sorted(v for v in values if v is not None and v > 0)

    if not usable:
        return None, 0.0

    best: list[float] = []

    for centre in usable:
        group = [
            value
            for value in usable
            if abs(value - centre) / max(centre, 1e-9) <= tolerance
        ]
        if len(group) > len(best):
            best = group

    if not best:
        return None, 0.0

    middle = len(best) // 2
    median = (
        best[middle]
        if len(best) % 2
        else 0.5 * (best[middle - 1] + best[middle])
    )

    return median, len(best) / len(usable)


def within_tolerance(prediction: float | None, gold: float, tolerance: float = TOLERANCE) -> bool:
    if prediction is None or gold == 0:
        return False
    return abs(prediction - gold) / abs(gold) <= tolerance


def format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))

    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


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


@dataclass(frozen=True)
class SetSpec:
    name: str
    question: str
    guidance: str
    allows_empty: bool
    max_answers: int

    min_support: float = 0.4


SET_SPECS: dict[str, SetSpec] = {
    "awardWonBy": SetSpec(
        name="awardWonBy",
        question="Who or what has received {subject}?",
        guidance=(
            "List the recipients: people or organizations that received it. "
            "Do not list presenters, founders, ceremonies, countries or years. "
            "Use one canonical name per recipient. List as many genuine "
            "recipients as you can recall, up to 60."
        ),
        allows_empty=False,
        max_answers=120,
        min_support=0.05,
    ),
    "companyTradesAtStockExchange": SetSpec(
        name="companyTradesAtStockExchange",
        question="At which stock exchanges is {subject} listed?",
        guidance=(
            "List stock exchange names, not ticker symbols and not stock "
            "indices. Do not transfer a parent company's listing to an "
            "unlisted subsidiary or brand. If this exact entity has no "
            "separately traded security, answer with an empty list."
        ),
        allows_empty=True,
        max_answers=6,
    ),
    "countryLandBordersCountry": SetSpec(
        name="countryLandBordersCountry",
        question="Which countries share a land border with {subject}?",
        guidance=(
            "List every country sharing a land border. Exclude countries "
            "separated only by sea. An island country with no land neighbours "
            "correctly has an empty list."
        ),
        allows_empty=True,
        max_answers=20,
    ),
    "personHasCityOfDeath": SetSpec(
        name="personHasCityOfDeath",
        question="In which city did {subject} die?",
        guidance=(
            "Give the city where this person died. Not their birthplace, not "
            "where they lived, not where they are buried. If this person was "
            "still alive on 1 July 2026, answer with an empty list."
        ),
        allows_empty=True,
        max_answers=1,
    ),
}


SET_SYSTEM = (
    "You are a precise knowledge base. You answer from memory with canonical "
    "names. The knowledge cutoff is 1 July 2026. An empty list means the "
    "relation genuinely has no objects -- never use it to mean you are "
    "unsure. Never invent plausible-sounding entries to pad an answer."
)


def _tokens(text: str) -> set[str]:
    return {t for t in normalize(text).split() if len(t) > 2}


def retrieve_examples(
    spec: SetSpec,
    train_rows: Sequence[dict],
    subject: str,
    *,
    n: int = 6,
) -> list[dict]:
    pool = [
        row
        for row in train_rows
        if row["Relation"] == spec.name
        and normalize(row["SubjectEntity"]) != normalize(subject)
    ]

    if not pool:
        return []

    target = _tokens(subject)

    def score(row: dict) -> float:
        other = _tokens(row["SubjectEntity"])
        if not target or not other:
            return 0.0
        return len(target & other) / len(target | other)

    ranked = sorted(pool, key=score, reverse=True)

    if not spec.allows_empty:
        return ranked[:n]

    empties = [r for r in ranked if not r["ObjectEntities"]]
    nonempties = [r for r in ranked if r["ObjectEntities"]]

    rate = len(empties) / len(pool)
    n_empty = min(len(empties), max(1, round(n * rate)), n - 1)

    chosen = empties[:n_empty] + nonempties[: n - n_empty]
    return chosen[:n]


def build_prompt(
    spec: SetSpec,
    subject: str,
    examples: Sequence[dict],
) -> str:
    parts = [spec.guidance]

    if examples:
        lines = []
        for row in examples:
            labels = [
                (group[0] if isinstance(group, list) else group)
                for group in row["ObjectEntities"]
            ]
            shown = labels[: min(spec.max_answers, 12)]
            lines.append(
                f"Q: {spec.question.format(subject=row['SubjectEntity'])}\n"
                f"A: {json.dumps(shown, ensure_ascii=False)}"
            )
        parts.append("Examples:\n\n" + "\n\n".join(lines))

    parts.append(
        f"Q: {spec.question.format(subject=subject)}\n"
        'A: Return JSON {"answers": [...]} and nothing else.'
    )

    return "\n\n".join(parts)


def parse_answers(payload) -> list[str]:
    if isinstance(payload, dict):
        payload = payload.get("answers", [])

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if isinstance(payload, dict):
            payload = payload.get("answers", [])

    if not isinstance(payload, list):
        return []

    out: list[str] = []
    for item in payload:
        if isinstance(item, list):
            item = item[0] if item else None
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def aggregate(
    samples: Sequence[Sequence[str]],
    spec: SetSpec,
    *,
    min_support: float | None = None,
) -> list[str]:
    if not samples:
        return []

    support = spec.min_support if min_support is None else min_support

    counts: Counter[str] = Counter()
    display: dict[str, str] = {}

    for sample in samples:
        seen: set[str] = set()
        for answer in sample:
            key = normalize(answer)
            if not key or key in seen:
                continue
            seen.add(key)
            counts[key] += 1
            display.setdefault(key, answer)

    if not counts:
        return []

    total = len(samples)
    kept = [
        (key, count)
        for key, count in counts.items()
        if count / total >= support
    ]
    kept.sort(key=lambda item: (-item[1], display[item[0]]))

    if not kept and not spec.allows_empty:
        best = max(counts.items(), key=lambda item: item[1])
        kept = [best]

    return [display[key] for key, _ in kept[: spec.max_answers]]


GRID = [round(0.05 * i, 2) for i in range(1, 21)]

FLAG = {
    "awardWonBy": "--award-support",
    "companyTradesAtStockExchange": "--company-support",
    "countryLandBordersCountry": "--border-support",
    "personHasCityOfDeath": "--death-support",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gold", required=True, help="TRAIN gold jsonl")
    parser.add_argument("-s", "--samples", required=True, help="--dump-samples jsonl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    official = load_official()

    gold_rows = official.read_jsonl_file(args.gold)
    gold_by_key = {(r["SubjectEntity"], r["Relation"]): r for r in gold_rows}

    traces: dict[tuple[str, str], list[list[str]]] = {}
    for line in Path(args.samples).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            trace = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "samples" in trace:
            traces[(trace["SubjectEntity"], trace["Relation"])] = trace["samples"]

    if not traces:
        raise SystemExit("no traced set-relation rows found")

    by_relation: dict[str, list[tuple[str, str]]] = {}
    for key in traces:
        by_relation.setdefault(key[1], []).append(key)

    fitted: dict[str, float] = {}

    header = f"{'relation':<32} {'default':>8} {'fitted':>8} {'delta':>8}  support"
    print(header)
    print("-" * len(header))

    for relation in sorted(by_relation):
        spec = SET_SPECS[relation]
        keys = by_relation[relation]
        rows = [gold_by_key[k] for k in keys if k in gold_by_key]

        if not rows:
            continue

        def score(support: float) -> float:
            predictions = [
                {
                    "SubjectEntity": key[0],
                    "Relation": key[1],
                    "ObjectEntities": aggregate(
                        traces[key], spec, min_support=support
                    ),
                }
                for key in keys
                if key in gold_by_key
            ]
            per_row = official.evaluate_per_sr_pair(
                predictions, rows, official.RELATION_TYPE, tolerance=0.05
            )
            return sum(r["f1"] for r in per_row) / len(per_row)

        default_value = score(spec.min_support)
        best_support, best_value = spec.min_support, default_value

        for support in GRID:
            value = score(support)
            if value > best_value:
                best_support, best_value = support, value

        fitted[relation] = best_support
        print(
            f"{relation:<32} {default_value:>8.3f} {best_value:>8.3f} "
            f"{best_value - default_value:>+8.3f}  {spec.min_support:.2f} -> "
            f"{best_support:.2f}  (n={len(rows)})"
        )

    print("\nrun.py flags:")
    print("  " + " ".join(f"{FLAG[r]} {v}" for r, v in fitted.items() if r in FLAG))

    print("\nas environment variables:")
    env = {
        "awardWonBy": "AWARD_SUPPORT",
        "companyTradesAtStockExchange": "COMPANY_SUPPORT",
        "countryLandBordersCountry": "BORDER_SUPPORT",
        "personHasCityOfDeath": "DEATH_SUPPORT",
    }
    print("  " + " ".join(f"{env[r]}={v}" for r, v in fitted.items() if r in env))

    if args.out:
        Path(args.out).write_text(json.dumps(fitted, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
