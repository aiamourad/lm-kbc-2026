# Unit tests for the aggregation, recurrence and deduplication rules.
from __future__ import annotations

import json

from aggregation import (SET_SPECS, SQ_MILE_IN_SQ_KM, aggregate, build_prompt,
                         cluster_vote, format_number, normalize, parse_answers,
                         parse_number, retrieve_examples, within_tolerance)
from scorer import find_official


def check_clustering(name, got, want):
    assert got == want, f"{name}: got {got!r} want {want!r}"
    print(f"  ok  {name}")

check_clustering("plain", parse_number("The area is 255.32 km2"), 255.32)

check_clustering("commas", parse_number("capacity of 74,879 spectators"), 74879.0)

check_clustering("spaces", parse_number("area 1 250 km2"), 1250.0)

check_clustering("leading text", parse_number("about 0.43 square kilometres"), 0.43)

check_clustering("none", parse_number("no idea"), None)

check_clustering("sq mi", round(parse_number("is 28.5 sq mi", detect_units=True), 4),
      round(28.5 * SQ_MILE_IN_SQ_KM, 4))

check_clustering("hectares", round(parse_number("is 500 hectares", detect_units=True), 4), 5.0)

check_clustering("unit far away ignored", parse_number("is 74.36 km2, or 28.7 sq mi", detect_units=True), 74.36)

value, support = cluster_vote([255.32, 255.0, 256.0, 909.0])

assert within_tolerance(value, 255.32), value

check_clustering("cluster support", round(support, 3), 0.75)

value, _ = cluster_vote([100.0, 900.0, 902.0, 905.0])

assert within_tolerance(value, 902.0), value

print("  ok  densest cluster wins over first-seen")

check_clustering("empty", cluster_vote([])[0], None)

check_clustering("drops nonpositive", cluster_vote([-5.0, 0.0, 42.0])[0], 42.0)

check_clustering("format int", format_number(74879.0), "74879")

check_clustering("format dec", format_number(0.43), "0.43")

print("all clustering tests passed")


def check_recurrence(name, got, want):
    assert got == want, f"{name}: got {got!r} want {want!r}"
    print(f"  ok  {name}")

border = SET_SPECS["countryLandBordersCountry"]

death = SET_SPECS["personHasCityOfDeath"]

award = SET_SPECS["awardWonBy"]

s = [["France","Spain"],["France","Spain"],["France","Andorra"],["France","Spain"]]

check_recurrence("frequent kept", aggregate(s, border, min_support=0.5), ["France","Spain"])

check_recurrence("noise dropped", "Andorra" in aggregate(s, border, min_support=0.5), False)

check_recurrence("empty when nothing recurs",
      aggregate([[],[],["Xland"],[]], border, min_support=0.5), [])

check_recurrence("empty samples -> empty", aggregate([[],[],[]], border, min_support=0.5), [])

out = aggregate([["Alice"],["Bob"],["Alice"]], award, min_support=0.9)

check_recurrence("non-empty relation always answers", out, ["Alice"])

check_recurrence("death capped to 1",
      len(aggregate([["Paris","Lyon"],["Paris","Lyon"],["Paris","Lyon"]], death,
                    min_support=0.5)), 1)

s2 = [["Kaua'i"],["Kauaʻi"],["Kauai"]]

check_recurrence("alias dedup", len(aggregate(s2, border, min_support=0.9)), 1)

check_recurrence("intra-sample dup ignored",
      aggregate([["France","France","France"],[ "Spain"]], border, min_support=0.9), [])

check_recurrence("parse dict", parse_answers({"answers":["A","B"]}), ["A","B"])

check_recurrence("parse list", parse_answers(["A","B"]), ["A","B"])

check_recurrence("parse json string", parse_answers('{"answers":["A"]}'), ["A"])

check_recurrence("parse nested", parse_answers({"answers":[["A"],["B"]]}), ["A","B"])

check_recurrence("parse junk", parse_answers("not json"), [])

train_file = find_official().parent / "data" / "train.jsonl"

train = [json.loads(l) for l in open(train_file)]

ex = retrieve_examples(SET_SPECS["companyTradesAtStockExchange"], train, "Some GmbH", n=6)

assert any(not r["ObjectEntities"] for r in ex), "no empty demonstration"

assert any(r["ObjectEntities"] for r in ex), "no non-empty demonstration"

print("  ok  retrieval stratifies empties")

assert all(normalize(r["SubjectEntity"]) != normalize("Some GmbH") for r in ex)

print("  ok  retrieval excludes target")

p = build_prompt(border, "Somalia", retrieve_examples(border, train, "Somalia", n=4))

assert "Somalia" in p and "Q:" in p and "answers" in p

print("  ok  prompt well-formed")

print("all sets tests passed")


from postprocess import alias_pairs, dedup

pairs = alias_pairs(str(find_official().parent / "data" / "train.jsonl"))

check_recurrence("alias pair attested", ("russia", "russian federation") in pairs, True)

check_recurrence("no transitive fusion",
      ("democratic republic of the congo", "republic of the congo") in pairs, False)

check_recurrence("drops second alias",
      dedup(["Latvia", "Russia", "Russian Federation"], pairs),
      ["Latvia", "Russia"])

check_recurrence("keeps distinct entities",
      dedup(["Democratic Republic of the Congo", "Republic of the Congo"], pairs),
      ["Democratic Republic of the Congo", "Republic of the Congo"])

check_recurrence("dedup never adds", len(dedup(["Russia", "Russian Federation"], pairs)) <= 2, True)

print("all alias tests passed")
