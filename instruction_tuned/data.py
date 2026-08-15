# Dataset loading, relation specs, and split handling.
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "dataset2026-main" / "data"

RELATION_TYPE = {
    "awardWonBy": "string",
    "hasCapacity": "numeric",
    "hasArea": "numeric",
    "countryLandBordersCountry": "string",
    "personHasCityOfDeath": "string",
    "companyTradesAtStockExchange": "string",
}


@dataclass(frozen=True)
class RelationSpec:

    name: str
    value_type: str
    question: str
    scope: str
    unit: str = ""
    allows_empty: bool = True
    empty_rate: float = 0.0
    typical_size: int = 1
    n_samples: int = 8
    max_candidates: int = 40
    max_tokens: int = 1536
    answers_per_call: int = 0
    identify_first: bool = False
    pitfalls: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_numeric(self) -> bool:
        return self.value_type == "numeric"


RELATIONS: dict[str, RelationSpec] = {
    "countryLandBordersCountry": RelationSpec(
        name="countryLandBordersCountry",
        value_type="string",
        question="Which countries share a land border with {subject}?",
        scope=(
            "Countries (or comparable territories) that share a LAND border with the "
            "subject. Maritime borders are excluded, for example Russia-Japan and "
            "Samoa-USA do not count. An island country with no land border has an "
            "EMPTY answer set. Only currently-recognised states count; deprecated or "
            "disputed border statements do not. A border through a country's INTEGRAL "
            "overseas territory counts (France-Brazil and France-Suriname via French "
            "Guiana; Spain-Morocco via Ceuta and Melilla). A border via a "
            "non-integral dependency does NOT count (Cyprus-United Kingdom via the "
            "Sovereign Base Areas). Enclave borders count (Vatican City-Italy, "
            "San Marino-Italy)."
        ),
        allows_empty=True,
        empty_rate=0.18,
        typical_size=3,
        n_samples=8,
        max_candidates=24,
        max_tokens=1024,
        pitfalls=(
            "Island nations (New Zealand, Iceland, Japan, Sri Lanka, Madagascar, "
            "Cuba, Mauritius) have an EMPTY answer set -- do not list nearby countries.",
            "List every neighbour, not just the famous ones. Missing neighbours costs "
            "as much as inventing them.",
            "Do not include the subject country itself.",
        ),
    ),
    "personHasCityOfDeath": RelationSpec(
        name="personHasCityOfDeath",
        value_type="string",
        question="In which city did {subject} die?",
        scope=(
            "The CITY where the person died -- city granularity, not the country, "
            "state or region. If the person is STILL ALIVE, or no locality is "
            "publicly known, the answer set is EMPTY. At most one city."
        ),
        allows_empty=True,
        empty_rate=0.42,
        typical_size=1,
        n_samples=8,
        max_candidates=8,
        max_tokens=768,
        pitfalls=(
            "About 40% of these people are still alive as of 1 July 2026. If the "
            "person is alive, or you are not confident they have died, the answer is "
            "EMPTY. Guessing a city for a living person scores zero.",
            "Answer with the city only: 'Los Angeles', not 'Los Angeles, California' "
            "and not 'United States'.",
            "Use the municipality where death occurred, which is not always the city "
            "the person is associated with.",
        ),
    ),
    "hasCapacity": RelationSpec(
        name="hasCapacity",
        value_type="numeric",
        question="What is the maximum spectator capacity of {subject}?",
        scope=(
            "The MAXIMUM spectator capacity of the venue, as an integer number of "
            "people (Wikidata P1083). Where several capacities exist -- seated versus "
            "total, before versus after a renovation -- the HIGHEST published figure "
            "is used. Exactly one number."
        ),
        unit="people",
        identify_first=False,
        allows_empty=False,
        empty_rate=0.0,
        typical_size=1,
        n_samples=8,
        max_candidates=12,
        max_tokens=768,
        pitfalls=(
            "The subject string carries a disambiguating location ('X in Jiaxing'). "
            "Use it to identify the right venue; do not answer about a same-named "
            "venue elsewhere.",
            "Report the highest published capacity, not the current seated-only "
            "configuration.",
            "A bare integer with no thousands separators, units or ranges.",
        ),
    ),
    "awardWonBy": RelationSpec(
        name="awardWonBy",
        value_type="string",
        question="Who has won the {subject}?",
        scope=(
            "Entities that have received the SPECIFIC award named by the subject. "
            "Winners are the RECIPIENT entities -- people, groups, organizations, "
            "projects -- never the winning works. A predecessor or successor award is "
            "a DISTINCT award and its winners do not count (the Medal of Freedom is "
            "not the Presidential Medal of Freedom). Rescinded awards are excluded. "
            "Gold sets run to hundreds of recipients."
        ),
        allows_empty=False,
        empty_rate=0.0,
        typical_size=60,
        n_samples=16,
        max_candidates=400,
        answers_per_call=40,
        max_tokens=8192,
        pitfalls=(
            "Gold sets here are large -- often 50 to 600 recipients covering every "
            "edition of the award. Recall is the binding constraint: enumerate "
            "exhaustively, working through the award's history year by year.",
            "Name the person or organization that received the award, never the book, "
            "film, album or paper that won it.",
            "Do not include winners of a similarly-named but distinct award.",
        ),
    ),
    "companyTradesAtStockExchange": RelationSpec(
        name="companyTradesAtStockExchange",
        value_type="string",
        question="On which stock exchanges are shares of {subject} traded?",
        scope=(
            "The stock exchange(s) on which the company's shares are publicly traded, "
            "as of 1 July 2026. Multiple listings are possible. A subsidiary that is "
            "not separately listed, a private company, a delisted company, and a "
            "trade association all have an EMPTY answer set."
        ),
        allows_empty=True,
        empty_rate=0.34,
        typical_size=1,
        n_samples=8,
        max_candidates=10,
        max_tokens=768,
        pitfalls=(
            "About a third of these subjects are not listed at all -- private "
            "companies, wholly-owned subsidiaries, trade associations, mutuals, "
            "state-owned firms, and companies taken private or delisted before "
            "1 July 2026. Those are EMPTY.",
            "Name the exchange, not the ticker: 'New York Stock Exchange', not 'AAPL'.",
            "Listing status is as of 1 July 2026: exclude exchanges the company has "
            "since delisted from, include listings added up to that date.",
        ),
    ),
    "hasArea": RelationSpec(
        name="hasArea",
        value_type="numeric",
        question="What is the area of {subject} in square kilometres?",
        scope=(
            "The surface area of the geographic entity in SQUARE KILOMETRES (km2). "
            "For countries this is the TOTAL area, land plus inland water "
            "(Wikidata P2046 preferred rank). Areas published in hectares, square "
            "miles or acres are converted to km2. Exactly one number."
        ),
        unit="square kilometres",
        allows_empty=False,
        empty_rate=0.0,
        typical_size=1,
        n_samples=8,
        max_candidates=12,
        max_tokens=768,
        pitfalls=(
            "Answer in square kilometres. A square-miles figure is wrong by 2.59x and "
            "a hectares figure by 100x -- both far outside the 5% tolerance.",
            "For a country use total area including inland water, not land area.",
            "Small islands and lakes need a precise figure, not a rounded order of "
            "magnitude: tolerance is only 5%.",
        ),
    ),
}


@dataclass(frozen=True)
class Row:

    subject: str
    relation: str
    gold: tuple[tuple[str, ...], ...] = ()

    @property
    def spec(self) -> RelationSpec:
        return RELATIONS[self.relation]

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject, self.relation)

    @property
    def gold_labels(self) -> list[str]:
        return [aliases[0] for aliases in self.gold if aliases]


def load_split(split: str, data_dir: Path | None = None) -> list[Row]:
    path = (data_dir or DATA_DIR) / f"{split}.jsonl"
    rows: list[Row] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            objects = record.get("ObjectEntities") or []
            gold = tuple(
                tuple(str(a) for a in aliases) if isinstance(aliases, list) else (str(aliases),)
                for aliases in objects
            )
            rows.append(
                Row(
                    subject=record["SubjectEntity"],
                    relation=record["Relation"],
                    gold=gold,
                )
            )
    return rows


def write_predictions(path: str | Path, predictions: dict[tuple[str, str], Sequence[str]],
                      rows: Sequence[Row]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            objects = list(predictions.get(row.key, []))
            handle.write(
                json.dumps(
                    {
                        "SubjectEntity": row.subject,
                        "Relation": row.relation,
                        "ObjectEntities": objects,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def iter_by_relation(rows: Sequence[Row]) -> Iterator[tuple[str, list[Row]]]:
    for relation in RELATIONS:
        subset = [r for r in rows if r.relation == relation]
        if subset:
            yield relation, subset
