# Proposing candidates: approach instructions, constrained decoding, set expansion.
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
from dataclasses import asdict, field
from pathlib import Path
from typing import Any, Sequence

from .clients import OpenAICompatibleClient
from .data import RELATIONS, RelationSpec, Row
from .group import (
    Candidate,
    cluster_numbers,
    cluster_strings,
    normalize_string,
    parse_number,
)
from .decode import (
    EXCLUSIVE,
    INDEPENDENT,
    Calibrator,
    DecodeResult,
    decode,
    geometric_missing_mass,
)


PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answers": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "answers"],
    "additionalProperties": False,
}

COUNT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "count": {"type": "integer"},
    },
    "required": ["reasoning", "count"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are a precise knowledge base engineer. You answer strictly from your own "
    "parametric knowledge -- you have no access to search or external references. "
    "Your answers populate a knowledge base, so they must be complete and correct: "
    "a missing object is as costly as an invented one. When you do not know, say so "
    "by returning an empty answer set rather than guessing a plausible-looking "
    "entity. The knowledge base describes the world as of 1 July 2026."
)

def _scope_block(spec: RelationSpec) -> str:
    rules = "\n".join(f"- {p}" for p in spec.pitfalls)
    return (
        f"RELATION: {spec.name}\n"
        f"DEFINITION: {spec.scope}\n"
        f"RULES:\n{rules}"
    )

def few_shot_examples(
    spec: RelationSpec,
    train_rows: Sequence[Row],
    *,
    n: int = 6,
    seed: int = 0,
) -> list[Row]:
    pool = [r for r in train_rows if r.relation == spec.name]
    if not pool:
        return []
    rng = random.Random(seed)
    empties = [r for r in pool if not r.gold]
    nonempties = [r for r in pool if r.gold]

    if not spec.allows_empty or not empties:
        return rng.sample(nonempties or pool, min(n, len(nonempties or pool)))

    n_empty = max(1, round(n * spec.empty_rate))
    n_empty = min(n_empty, len(empties), n - 1)
    chosen = rng.sample(empties, n_empty) + rng.sample(
        nonempties, min(n - n_empty, len(nonempties))
    )
    rng.shuffle(chosen)
    return chosen

def _format_example(row: Row, spec: RelationSpec) -> str:
    if row.gold:
        limit = 8 if spec.answers_per_call else 40
        answers = ", ".join(f'"{label}"' for label in row.gold_labels[:limit])
        if len(row.gold_labels) > limit:
            answers += f", ...   ({len(row.gold_labels)} recipients in total)"
        body = f"[{answers}]"
    else:
        body = "[]   <- empty: no object satisfies the relation"
    return f"Subject: {row.subject}\nAnswers: {body}"

def propose_prompt(
    row: Row,
    spec: RelationSpec,
    *,
    shots: Sequence[Row] = (),
    strategy: str = "",
) -> str:
    parts = [_scope_block(spec)]

    if shots:
        examples = "\n\n".join(_format_example(r, spec) for r in shots)
        parts.append(f"EXAMPLES OF CORRECT KNOWLEDGE BASE ENTRIES:\n\n{examples}")

    question = spec.question.format(subject=row.subject)
    ask = [f"NOW ANSWER FOR THIS SUBJECT.\n\nSubject: {row.subject}\nQuestion: {question}"]

    if spec.is_numeric and spec.identify_first:
        ask.append(
            "In 'reasoning', first pin down exactly which venue this is: the "
            "city and country, the sport, the club or institution that uses it, "
            "and roughly when it was built. Only then give the figure. If you "
            "cannot identify the specific venue, say so in 'reasoning' and give "
            "your single best estimate anyway.\n"
            f"Then give exactly one number, in {spec.unit}, as a bare figure "
            "with no separators, units, ranges or approximations."
        )
    elif spec.is_numeric:
        ask.append(
            f"Give exactly one number, in {spec.unit}, as a bare figure with no "
            "separators, units, ranges or approximations."
        )
    elif spec.allows_empty:
        ask.append(
            "Return every object that satisfies the definition. If none does, return "
            "an empty list -- that is a correct and expected answer here, not a failure."
        )
    elif spec.answers_per_call:
        ask.append(
            f"Name up to {spec.answers_per_call} objects you are confident about "
            "-- not an exhaustive list. A short, correct, COMPLETE JSON reply is "
            "worth far more than a long one that gets cut off: a truncated reply "
            "cannot be parsed and is discarded entirely. Stop well before that."
        )
    else:
        ask.append("Return every object that satisfies the definition.")

    if strategy:
        ask.append(f"APPROACH: {strategy}")

    ask.append(
        "First think briefly in 'reasoning', then give the final objects in 'answers'. "
        "Each answer must be the entity's own name, with no explanation attached."
    )
    parts.append("\n\n".join(ask))
    return "\n\n".join(parts)

def verify_prompt(row: Row, spec: RelationSpec, candidate: str) -> str:
    question = spec.question.format(subject=row.subject)
    if spec.is_numeric:
        claim = (
            f"Claim: the correct answer to \"{question}\" is {candidate} "
            f"{spec.unit}, to within 5%."
        )
    else:
        claim = f'Claim: "{candidate}" is one of the correct answers to "{question}"'

    return (
        f"{_scope_block(spec)}\n\n"
        f"{claim}\n\n"
        "Judge this claim against the definition above, using only your own knowledge "
        "of the world as of 1 July 2026. Answer with exactly one word, Yes or No.\n"
        "Answer:"
    )

def count_prompt(row: Row, spec: RelationSpec) -> str:
    question = spec.question.format(subject=row.subject)
    return (
        f"{_scope_block(spec)}\n\n"
        f"Subject: {row.subject}\n"
        f"Question: {question}\n\n"
        "Do not list the objects. Estimate only HOW MANY objects satisfy this "
        "relation in total. For an award, that is the total number of distinct "
        "recipients across every edition of the award since it was founded. "
        "Reason briefly about the award's founding year, how often it is given and "
        "how many recipients each edition has, then give a single integer."
    )

STRATEGIES: dict[str, tuple[str, ...]] = {
    "countryLandBordersCountry": (
        "Walk the subject's land frontier clockwise, naming each country you cross into.",
        "Recall the subject's neighbours by region, then check each for a genuine land "
        "border rather than a sea crossing.",
        "First decide whether the subject is an island or landlocked, then enumerate "
        "accordingly.",
        "Name the countries you are confident about, then deliberately search for "
        "easily-forgotten small or enclaved neighbours and overseas-territory borders.",
    ),
    "personHasCityOfDeath": (
        "First establish whether this person is alive or dead as of 1 July 2026, and "
        "only then, if dead, recall where they died.",
        "Recall the person's identity, field and dates, then their place of death.",
        "Recall the circumstances of the death -- illness, accident, age -- and the "
        "city where it occurred.",
        "Consider whether you are confusing this person with a similarly-named one, "
        "then answer.",
    ),
    "hasCapacity": (
        "Identify the specific venue from its location qualifier, then recall its "
        "published maximum capacity.",
        "Recall the venue's tier and typical use -- top-flight stadium, college ground, "
        "arena -- and the capacity range that implies, then commit to a figure.",
        "Recall the capacity at its largest configuration, including any expansion.",
        "Compare the venue against a stadium you know precisely in the same city or "
        "league, then adjust.",
    ),
    "hasArea": (
        "Recall the published total area of this entity directly, in square kilometres.",
        "Estimate from the entity's dimensions -- length and width -- then convert to "
        "square kilometres.",
        "Compare against a geographic entity whose area you know precisely, then scale.",
        "Recall the figure in whatever unit you know it best, then convert carefully "
        "to square kilometres.",
    ),
    "awardWonBy": (
        "Work through the award's history chronologically by decade, naming recipients "
        "in each period.",
        "Name the most famous recipients first, then work outwards to the less "
        "celebrated ones.",
        "Recall the award's founding, its frequency and how many recipients per "
        "edition, then enumerate edition by edition.",
        "Group recipients by nationality or field, and enumerate within each group.",
    ),
    "companyTradesAtStockExchange": (
        "First establish whether this entity is publicly listed at all as of 1 July "
        "2026, then name the exchanges.",
        "Recall the company's home country and its primary listing there, then check "
        "for secondary or cross-listings.",
        "Consider whether the entity is a subsidiary, a private company, a mutual or a "
        "trade association -- any of which is not separately listed.",
        "Recall the company's ticker symbols and the exchange each belongs to.",
    ),
}

def strategies_for(relation: str, n: int) -> list[str]:
    pool = STRATEGIES.get(relation) or ("",)
    return [pool[i % len(pool)] for i in range(n)]


logger = logging.getLogger("lmkbc.solver")

_EXCLUSIVE_RELATIONS = {"personHasCityOfDeath", "hasCapacity", "hasArea"}

class SolverConfig:

    n_samples: int = 0
    temperature: float = 0.8
    n_shots: int = 6
    verify: bool = True
    estimate_count: bool = True
    max_verify: int = 60
    missing_mass_cap: float = 1.0
    seed: int = 0

    def samples_for(self, spec: RelationSpec) -> int:
        return self.n_samples or spec.n_samples

class RowPrediction:

    subject: str
    relation: str
    objects: list[str]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    expected_f1: float = 0.0
    k: int = 0
    margin: float = 0.0
    n_proposed: int = 0
    count_estimate: float | None = None
    n_samples_ok: int = 0
    error: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject, self.relation)

class Solver:

    def __init__(
        self,
        client: OpenAICompatibleClient,
        train_rows: Sequence[Row],
        config: SolverConfig | None = None,
        calibrator: Calibrator | None = None,
        verifier: OpenAICompatibleClient | None = None,
        proposers: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.train_rows = list(train_rows)
        self.config = config or SolverConfig()
        self.calibrator = calibrator or Calibrator.default()
        self.verifier = verifier or client
        self.proposers = proposers or {}

    async def propose(self, row: Row, spec: RelationSpec) -> tuple[list[Candidate], int]:
        n_samples = self.config.samples_for(spec)
        shots = few_shot_examples(
            spec, self.train_rows, n=self.config.n_shots, seed=self.config.seed
        )
        strategies = strategies_for(spec.name, min(4, n_samples))
        per_strategy = max(1, math.ceil(n_samples / len(strategies)))

        async def one(strategy: str, index: int) -> list[list[str]]:
            prompt = propose_prompt(row, spec, shots=shots, strategy=strategy)
            result = await self.client.complete(
                system=SYSTEM,
                prompt=prompt,
                n=per_strategy,
                temperature=self.config.temperature,
                max_tokens=spec.max_tokens,
                schema=PROPOSE_SCHEMA,
                schema_name="proposal",
                nonce=f"{self.config.seed}:{index}",
            )
            out = []
            for completion in result.completions:
                data = completion.data
                if not isinstance(data, dict):
                    continue
                answers = data.get("answers")
                if not isinstance(answers, list):
                    continue
                gate = str(data.get("gate") or "").strip().upper()
                out.append(
                    (
                        [str(a) for a in answers if isinstance(a, (str, int, float))],
                        gate == "HELD",
                    )
                )
            return out

        batches = await asyncio.gather(
            *(one(s, i) for i, s in enumerate(strategies)), return_exceptions=True
        )
        pairs: list[tuple[list[str], bool]] = []
        for batch in batches:
            if isinstance(batch, Exception):
                logger.warning("proposal batch failed for %s: %s", row.subject, batch)
                continue
            pairs.extend(batch)

        if not pairs:
            return [], 0

        samples = [answers for answers, _ in pairs]
        candidates = (
            cluster_numbers(samples) if spec.is_numeric else cluster_strings(samples)
        )
        candidates = candidates[: spec.max_candidates]
        _attach_gate(candidates, pairs, numeric=spec.is_numeric)
        return candidates, len(samples)

    async def verify(
        self, row: Row, spec: RelationSpec, candidates: Sequence[Candidate]
    ) -> None:
        targets = list(candidates)[: self.config.max_verify]

        async def one(candidate: Candidate) -> None:
            prompt = verify_prompt(row, spec, candidate.label)
            try:
                distribution = await self.verifier.complete_logprobs(
                    system=SYSTEM, prompt=prompt, top_k=20, max_tokens=2
                )
            except Exception as exc:
                logger.warning("verify failed for %r: %s", candidate.label, exc)
                distribution = {}
            candidate.features["verification"] = _yes_probability(distribution)

        await asyncio.gather(*(one(c) for c in targets))
        for candidate in candidates[self.config.max_verify :]:
            candidate.features.setdefault("verification", 0.5)

    async def estimate_count(self, row: Row, spec: RelationSpec) -> float | None:
        result = await self.client.complete(
            system=SYSTEM,
            prompt=count_prompt(row, spec),
            n=3,
            temperature=0.7,
            max_tokens=1024,
            schema=COUNT_SCHEMA,
            schema_name="count",
            nonce=f"count:{self.config.seed}",
        )
        values = []
        for completion in result.completions:
            data = completion.data
            if isinstance(data, dict):
                parsed = parse_number(str(data.get("count")))
                if parsed is not None and parsed >= 0:
                    values.append(parsed)
        if not values:
            return None
        return sorted(values)[len(values) // 2]

    def decode_row(
        self,
        row: Row,
        spec: RelationSpec,
        candidates: Sequence[Candidate],
        count_estimate: float | None,
    ) -> DecodeResult:
        ordered = sorted(candidates, key=lambda c: -c.support)
        for rank, candidate in enumerate(ordered):
            candidate.features.setdefault("verification", 0.5)
            candidate.features["recurrence"] = candidate.smoothed_frequency
            candidate.features["rank_fraction"] = 1.0 / (1.0 + rank)
            candidate.features["pool_fraction"] = 1.0 / (1.0 + len(ordered))
            candidate.prob = self.calibrator.probability(spec.name, candidate.features)

        mode = EXCLUSIVE if spec.name in _EXCLUSIVE_RELATIONS else INDEPENDENT
        missing = None
        if mode == INDEPENDENT and count_estimate is not None:
            expected_present = sum(c.prob for c in candidates)
            shortfall = max(0.0, count_estimate - expected_present)
            if self.config.missing_mass_cap > 0:
                missing = geometric_missing_mass(
                    min(shortfall, self.config.missing_mass_cap * expected_present)
                )

        return decode(
            [c.label for c in candidates],
            [c.prob for c in candidates],
            mode=mode,
            missing_mass=missing,
            gold_is_nonempty=not spec.allows_empty,
            max_k=spec.max_candidates,
        )

    async def solve_row(self, row: Row) -> RowPrediction:
        spec = RELATIONS[row.relation]
        proposer = self.proposers.get(row.relation)
        try:
            if proposer is not None:
                candidates, n_ok = await proposer.propose(row, spec)
            else:
                candidates, n_ok = await self.propose(row, spec)
        except Exception as exc:
            logger.exception("proposal failed for %s/%s", row.subject, row.relation)
            return RowPrediction(
                subject=row.subject, relation=row.relation, objects=[], error=str(exc)
            )

        if not candidates:
            return RowPrediction(
                subject=row.subject,
                relation=row.relation,
                objects=[],
                n_samples_ok=n_ok,
                error="no candidates proposed",
            )

        count_estimate = None
        tasks = []
        if self.config.verify:
            tasks.append(self.verify(row, spec, candidates))
        want_count = self.config.estimate_count and spec.typical_size > 5
        if want_count:
            tasks.append(self.estimate_count(row, spec))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            if want_count:
                tail = results[-1]
                count_estimate = tail if isinstance(tail, (int, float)) else None

        result = self.decode_row(row, spec, candidates, count_estimate)
        return RowPrediction(
            subject=row.subject,
            relation=row.relation,
            objects=result.selected,
            candidates=[_candidate_record(c) for c in candidates],
            expected_f1=result.expected_f1,
            k=result.k,
            margin=result.margin,
            n_proposed=len(candidates),
            count_estimate=count_estimate,
            n_samples_ok=n_ok,
        )

    async def solve(
        self, rows: Sequence[Row], *, progress_every: int = 25
    ) -> list[RowPrediction]:
        done = 0
        total = len(rows)

        async def one(row: Row) -> RowPrediction:
            nonlocal done
            prediction = await self.solve_row(row)
            done += 1
            if progress_every and done % progress_every == 0:
                logger.info("  %d/%d rows", done, total)
            return prediction

        return list(await asyncio.gather(*(one(r) for r in rows)))

def _attach_gate(
    candidates: Sequence[Candidate],
    pairs: Sequence[tuple[list[str], bool]],
    *,
    numeric: bool,
) -> None:
    from .group import normalize_string, parse_number, within_tolerance

    for candidate in candidates:
        held = total = 0
        key = normalize_string(candidate.label)
        value = candidate.value if numeric else None
        for answers, is_held in pairs:
            matched = False
            for answer in answers:
                if numeric:
                    parsed = parse_number(answer)
                    if parsed and value and within_tolerance(parsed, value):
                        matched = True
                        break
                elif normalize_string(answer) == key:
                    matched = True
                    break
            if matched:
                total += 1
                held += int(is_held)

def _candidate_record(candidate: Candidate) -> dict[str, Any]:
    record = asdict(candidate)
    record["variants"] = record["variants"][:4]
    return record

def _yes_probability(distribution: dict[str, float]) -> float:
    if not distribution:
        return 0.5
    yes = no = 0.0
    for token, logprob in distribution.items():
        stripped = token.strip().lower().lstrip('"\'*_ ')
        probability = math.exp(logprob)
        if stripped.startswith("yes") or stripped in ("y", "true", "correct"):
            yes += probability
        elif stripped.startswith("no") or stripped in ("n", "false", "incorrect"):
            no += probability
    total = yes + no
    if total <= 0:
        return 0.5
    return min(max(yes / total, 1e-4), 1 - 1e-4)


logger = logging.getLogger("lmkbc.structured")

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

OUTLINE = {
    "compact": 0.70,
    "even": 0.55,
    "elongated": 0.45,
    "pinched": 0.35,
    "branching": 0.25,
}

VENUE_CLASS = {
    "national_stadium": (45_000, 90_000),
    "cricket_test": (25_000, 100_000),
    "college_major": (30_000, 100_000),
    "topflight_football": (20_000, 55_000),
    "baseball_major": (35_000, 55_000),
    "midsize_club": (10_000, 25_000),
    "indoor_arena": (10_000, 20_000),
    "college_small": (3_000, 15_000),
    "small_ground": (1_000, 10_000),
    "minor_venue": (500, 5_000),
}

class FieldBlock:

    fields: dict[str, str] = field(default_factory=dict)
    answers: list[str] = field(default_factory=list)
    raw: str = ""

    def get(self, key: str) -> str | None:
        value = self.fields.get(key.upper())
        if value is None:
            return None
        value = value.strip()
        return None if value.upper() in ("NONE", "N/A", "") else value

    def number(self, key: str) -> float | None:
        value = self.get(key)
        if value is None:
            return None
        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)
        if not match:
            return None
        try:
            return float(match.group(0).replace(",", ""))
        except ValueError:
            return None

_FIELD_RE = re.compile(r"^\s*([A-Z_]{3,20})\s*:\s*(.*)$")

def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""

def build_system(relation: str) -> str:
    core = load_prompt("core_system_prompt")
    module = load_prompt(relation)
    if not core or not module:
        raise FileNotFoundError(f"missing prompt files for {relation} in {PROMPT_DIR}")
    return f"{core}\n\n{'=' * 70}\n\n{module}"

def build_user(subject: str, relation: str) -> str:
    template = load_prompt("userMessageTemplate") or "SubjectEntity: {subject}\nRelation: {relation}"
    return template.format(subject=subject, relation=relation)

def parse(text: str) -> FieldBlock:
    block = FieldBlock(raw=text or "")
    if not text:
        return block

    for line in text.splitlines():
        match = _FIELD_RE.match(line)
        if match and not line.strip().startswith("{"):
            key, value = match.group(1), match.group(2)
            block.fields.setdefault(key, value)

    for line in reversed(text.strip().splitlines()):
        line = line.strip().strip("`")
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        objects = payload.get("ObjectEntities")
        if isinstance(objects, list):
            block.answers = [str(o) for o in objects if str(o).strip()]
            break
    return block

def area_from_dimensions(block: FieldBlock) -> float | None:
    length, width = block.number("LENGTH"), block.number("MAXWIDTH")
    if not length or not width or length <= 0 or width <= 0:
        return None
    name = (block.get("CLASS") or "even").lower()
    coefficient = OUTLINE.get(name, 0.55)
    return length * width * coefficient

def area_from_bracket(block: FieldBlock) -> float | None:
    low, high = block.number("BRACKET_LOW"), block.number("BRACKET_HIGH")
    position = block.number("POSITION")
    if not low or not high or position is None or low <= 0 or high <= 0:
        return None
    position = min(max(position, 0.0), 1.0)
    return math.exp(math.log(low) + position * (math.log(high) - math.log(low)))

def resolve_area(block: FieldBlock) -> tuple[float | None, str]:
    held = block.number("VALUE") if (block.get("GATE") or "").upper() == "HELD" else None
    geometric = area_from_dimensions(block)

    if held and geometric:
        ratio = max(held, geometric) / min(held, geometric)
        if ratio > 2.5:
            logger.info("HELD %.4g contradicts geometry %.4g; using geometry", held, geometric)
            return geometric, "geometry_over_held"
        return held, "held"
    if held:
        return held, "held"
    if geometric:
        return geometric, "geometry"
    bracket = area_from_bracket(block)
    if bracket:
        return bracket, "bracket"
    fallback = block.number("VALUE")
    return (fallback, "value_fallback") if fallback else (None, "none")

def resolve_capacity(block: FieldBlock) -> tuple[float | None, str]:
    value = block.number("VALUE")
    if value is None:
        return None, "none"
    name = (block.get("CLASS") or "").lower()
    bounds = VENUE_CLASS.get(name)
    if not bounds:
        return value, "unchecked"
    low, high = bounds
    if value < low * 0.5 or value > high * 2.0:
        midpoint = math.exp((math.log(low) + math.log(high)) / 2)
        logger.info("capacity %.0f outside class %s; using %.0f", value, name, midpoint)
        return midpoint, "class_midpoint"
    return value, "in_class"

def confidence(block: FieldBlock, relation: str) -> float | None:
    if relation == "personHasCityOfDeath":
        p_city = block.number("P_CITY")
        if p_city is not None:
            return min(max(p_city, 0.0), 1.0)
    gate = (block.get("GATE") or "").upper()
    if gate == "HELD":
        return 0.75
    if gate == "ESTIMATED":
        return 0.35
    return None

def extract(block: FieldBlock, relation: str) -> tuple[list[str], dict[str, Any]]:
    meta: dict[str, Any] = {"gate": block.get("GATE"), "route": None}

    if relation == "hasArea":
        value, route = resolve_area(block)
        meta["route"] = route
        return ([f"{value:.6g}"] if value else []), meta
    if relation == "hasCapacity":
        value, route = resolve_capacity(block)
        meta["route"] = route
        return ([str(int(round(value)))] if value else []), meta

    meta["route"] = "json_line"
    return list(block.answers), meta

class StructuredProposer:

    def __init__(self, client, n_samples: int = 8, temperature: float = 0.7) -> None:
        self.client = client
        self.n_samples = n_samples
        self.temperature = temperature
        self.last_stats: dict[str, Any] = {}

    async def propose(self, row, spec):
        from .group import cluster_numbers, cluster_strings

        system = build_system(spec.name)
        user = build_user(row.subject, spec.name)

        result = await self.client.complete(
            system=system,
            prompt=user,
            n=self.n_samples,
            temperature=self.temperature,
            max_tokens=spec.max_tokens,
            schema=None,
        )

        pairs: list[tuple[list[str], bool]] = []
        gates: dict[str, int] = {}
        for completion in result.completions:
            block = parse(completion.text)
            answers, meta = extract(block, spec.name)
            gate = (meta.get("gate") or "").upper()
            gates[gate or "NONE"] = gates.get(gate or "NONE", 0) + 1
            pairs.append((answers, gate == "HELD"))

        if not pairs:
            return [], 0

        samples = [answers for answers, _ in pairs]
        candidates = (
            cluster_numbers(samples) if spec.is_numeric else cluster_strings(samples)
        )
        candidates = candidates[: spec.max_candidates]
        _attach_gate(candidates, pairs, numeric=spec.is_numeric)

        self.last_stats = {"gates": gates, "candidates": len(candidates)}
        return candidates, len(samples)


logger = logging.getLogger("lmkbc.mcts")

class ExpansionNode:

    move: str
    visits: int = 0
    yielded: int = 0
    exhausted: bool = False
    seen: set[str] = field(default_factory=set)

    @property
    def value(self) -> float:
        if not self.visits:
            return 0.0
        return 1.0 - math.exp(-self.yielded / self.visits / 8.0)

DECADE_FALLBACK: list[str] = [
    "were awarded before 1950",
    *[f"were awarded in the {decade}s" for decade in range(1950, 2030, 10)],
    "are the most recent recipients",
]


class SetExpansionSearch:

    def __init__(
        self,
        client: OpenAICompatibleClient,
        *,
        iterations: int = 20,
        c_puct: float = 1.4,
        concurrency: int = 4,
        base: Any = None,
    ) -> None:
        self.client = client
        self.iterations = iterations
        self.c_puct = c_puct
        self.concurrency = concurrency
        self.base = base
        self.last_stats: dict[str, Any] = {}

    async def _seed_facets(self, row: Row, spec: RelationSpec) -> list[str]:
        schema = {
            "type": "object",
            "properties": {
                "facets": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["facets"],
            "additionalProperties": False,
        }
        prompt = (
            f"The award is: {row.subject}\n\n"
            f"Definition: {spec.scope}\n\n"
            "You will enumerate this award's recipients one slice at a time. "
            "Propose 8 to 14 slices that between them cover the whole set of "
            "recipients without much overlap. Good slices are time periods "
            "(founding decade through the present), fields, or regions -- choose "
            "whichever actually structures THIS award. Each slice must be a short "
            "noun phrase usable in the sentence 'list the recipients who ...'."
        )
        result = await self.client.complete(
            system=SYSTEM,
            prompt=prompt,
            n=2,
            temperature=0.5,
            max_tokens=1024,
            schema=schema,
            schema_name="facets",
        )
        for completion in result.completions:
            data = completion.data
            if isinstance(data, dict) and isinstance(data.get("facets"), list):
                facets = [str(f).strip() for f in data["facets"] if str(f).strip()]
                if facets:
                    return facets[:14]
        logger.info("facet seeding failed for %r; using decade fallback", row.subject)
        return DECADE_FALLBACK

    async def _expand(self, row: Row, spec: RelationSpec, node: ExpansionNode,
                      known: Sequence[str]) -> list[str]:
        schema = {
            "type": "object",
            "properties": {"answers": {"type": "array", "items": {"type": "string"}}},
            "required": ["answers"],
            "additionalProperties": False,
        }
        already = list(known)[-120:]
        prompt = (
            f"Award: {row.subject}\n\n"
            f"Definition: {spec.scope}\n\n"
            f"Slice to enumerate: {node.move}\n\n"
            + (
                "Recipients already collected (do NOT repeat these):\n"
                + ", ".join(already)
                + "\n\n"
                if already
                else ""
            )
            + f"List up to {spec.answers_per_call or 40} recipients of this award "
            "within this slice that are not already collected. Name the recipient "
            "entity -- the person, group or organization -- never the work that "
            "won. A complete short reply beats a long truncated one, which cannot "
            "be parsed and is discarded. Return an empty list if you know of no "
            "further recipients in this slice; that is a useful answer and better "
            "than padding with names you are unsure of."
        )
        result = await self.client.complete(
            system=SYSTEM,
            prompt=prompt,
            temperature=0.7,
            max_tokens=spec.max_tokens,
            schema=schema,
            schema_name="expansion",
            nonce=f"exp:{node.visits}",
        )
        for completion in result.completions:
            data = completion.data
            if isinstance(data, dict) and isinstance(data.get("answers"), list):
                return [str(a).strip() for a in data["answers"] if str(a).strip()]
        return []

    def _select(self, nodes: Sequence[ExpansionNode], total_visits: int) -> ExpansionNode | None:
        live = [n for n in nodes if not n.exhausted]
        if not live:
            return None
        unvisited = [n for n in live if n.visits == 0]
        if unvisited:
            return unvisited[0]
        log_total = math.log(max(total_visits, 2))
        return max(
            live, key=lambda n: n.value + self.c_puct * math.sqrt(log_total / n.visits)
        )

    async def propose(self, row: Row, spec: RelationSpec) -> tuple[list[Candidate], int]:
        base_candidates: list[Candidate] = []
        base_samples = 0
        if self.base is not None:
            base_candidates, base_samples = await self.base.propose(row, spec)

        facets = await self._seed_facets(row, spec)
        nodes = [ExpansionNode(move=facet) for facet in facets]
        collected: dict[str, str] = {
            normalize_string(c.label): c.label for c in base_candidates
        }
        support: dict[str, set[str]] = {}
        total_visits = 0

        for _ in range(self.iterations):
            batch: list[ExpansionNode] = []
            for _ in range(self.concurrency):
                node = self._select([n for n in nodes if n not in batch], total_visits)
                if node is None:
                    break
                batch.append(node)
            if not batch:
                break

            known = list(collected.values())
            results = await asyncio.gather(
                *(self._expand(row, spec, node, known) for node in batch),
                return_exceptions=True,
            )
            for node, names in zip(batch, results):
                node.visits += 1
                total_visits += 1
                if isinstance(names, Exception):
                    logger.warning("facet %r failed: %s", node.move, names)
                    continue
                fresh = 0
                for name in names:
                    key = normalize_string(name)
                    if not key:
                        continue
                    if key not in collected:
                        collected[key] = name
                        fresh += 1
                    support.setdefault(key, set()).add(node.move)
                    node.seen.add(key)
                node.yielded += fresh
                if fresh == 0:
                    node.exhausted = True

        n_facets = max(len(facets), 1)
        by_key = {normalize_string(c.label): c for c in base_candidates}
        candidates: list[Candidate] = []
        for key, surface in collected.items():
            facet_support = len(support.get(key, ()))
            facet_frequency = facet_support / n_facets
            existing = by_key.get(key)
            sample_frequency = existing.frequency if existing else 0.0
            frequency = max(facet_frequency, sample_frequency)
            candidates.append(
                Candidate(
                    label=existing.label if existing else surface,
                    variants=existing.variants if existing else [surface],
                    support=round(frequency * n_facets),
                    n_samples=n_facets,
                )
            )
        candidates.sort(key=lambda c: -c.support)

        self.last_stats = {
            "facets": len(facets),
            "expansions": total_visits,
            "exhausted": sum(1 for n in nodes if n.exhausted),
            "from_sampling": len(base_candidates),
            "collected": len(collected),
            "added_by_search": len(collected) - len(base_candidates),
        }
        logger.info("set expansion for %r: %s", row.subject, self.last_stats)
        return candidates[: spec.max_candidates], n_facets
