# The final system

`google/gemma-3-27b-pt` (27.43B) answering all six relations by text
completion. No chat template, no fine-tuning, no retrieval. Official **0.6836**.

## How a row is answered

**Never ask a question.** Each relation has a completion format: a few solved
training rows, then an unfinished one.

```
The total area of Iceland is 103000 km2.
The total area of Estonia is 45335 km2.
The total area of Corfu is
```

**Ask many times, keep what repeats.** The prompt is sampled 64–256 times. For
numbers, the answer is the median of the largest cluster agreeing within the
scorer's 5% tolerance. For sets, a candidate is kept when it appears in at
least a fraction *τ* of samples, fitted per relation on train. Nothing reaching
*τ* means the empty set, so abstention falls out of the same rule.

**Two relations need one more step.** `awardWonBy` sets run to hundreds of
names, so a real winner may appear three times in 64 samples and recurrence
cannot separate it from a hallucination; the model verifies those rare
candidates and confirmed ones are added back. For `personHasCityOfDeath` about
40% of subjects are alive and the model invents a plausible city for them
consistently enough that checking the city fails, so we ask whether the person
is dead at all.

Finally, a prediction is dropped when another already names the same entity —
"Russian Federation" beside "Russia" — since the scorer matches each gold
entity to at most one prediction.

## Files

| | |
|---|---|
| `final_system.py` | runs everything below, for one split |
| `completion_system.py` | formats, sampling, parsing |
| `aggregation.py` | clustering, recurrence, fitting *τ* |
| `postprocess.py` | rescue, existence check, alias dedup, apply |
| `knowledge_probe.py` | the probe that chose the model; not in the pipeline |
| `client.py` | vLLM client; raises rather than returning silent empties if the server dies |
| `scorer.py` | loads the organisers' `evaluate.py` |
| `fitted-supports.json` | the thresholds used |
| `tests.py` | unit tests, no GPU |

## Running it

```bash
python final_system.py --model google/gemma-3-27b-pt --split val  --out out/val.jsonl
python final_system.py --model google/gemma-3-27b-pt --split test --out out/test.jsonl
```

`--fit` refits the thresholds, which a different checkpoint needs.
`final_system.py` prints every command it runs, so a single stage is easiest
reproduced by copying one out of its output.

## One trap

Thresholds are tied to the sampling budget. Change `BUDGETS` and `tau` has to
be refitted with `--fit`; the old value is not approximately right. This also
sets the band the awards rescue works in — `--min-count / --samples` must stay
below the fitted `awardWonBy` threshold, or the stage silently does nothing.
