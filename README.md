# The Instruction Tax on Factual Recall: Base Models Can Beat Their Instruction-Tuned Counterparts

Our entry to the [LM-KBC 2026 shared task](https://lm-kbc.github.io/challenge2026/)
at AKBC. Official score **0.6836**; the task baseline is 0.2964.

## The task

Given a subject and a relation — *(Albert Einstein, city of death)* — return
every correct object: `["Princeton"]`. Often the answer is nothing at all, and
about a fifth of the evaluated rows are like that. Six relations: two numeric,
counted correct within 5%, and four set-valued, one of which can have hundreds
of answers.

No web search, no retrieval, no fine-tuning. Open weights only, 32B total.

## The finding

Use the **base** checkpoint and let it continue text:

```
The total area of Iceland is 103000 km2.
The total area of Corfu is
```

Its **instruction-tuned** sibling, asked the same thing through a chat
interface, scores 0.568 where the base model scores 0.684.

The gap is not the prompt. Give the tuned model the same completion prompt and
it recovers almost nothing. The right answer stops appearing in its samples at
all: 64 draws from the base model contain the correct figure 95% of the time,
the tuned model 69%. Chat training made facts harder to get back out. We call
this the **instruction tax** and measure it on four model families.

The system follows: prompt a base model with a few examples, sample each row
64–256 times, keep what recurs, answer nothing when nothing recurs.

## Contents

| | |
|---|---|
| [`base_completion/`](base_completion/) | the final system |
| [`instruction_tuned/`](instruction_tuned/) | the chat system it is compared against |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | each table, and the command that produces it |
| [`SUBMISSIONS.md`](SUBMISSIONS.md) | every submission, in order |

## Running it

Needs vLLM serving `gemma-3-27b-pt` (we used 2×H100), Python 3.10+, and the
task's dataset ([`lm-kbc/dataset2026`](https://github.com/lm-kbc/dataset2026))
checked out as `dataset2026-main/` beside this directory.

```bash
vllm serve google/gemma-3-27b-pt --port 8000 --tensor-parallel-size 2 \
    --max-model-len 16384 --dtype bfloat16 --enable-prefix-caching

cd base_completion
python final_system.py --model google/gemma-3-27b-pt --split val  --out out/val.jsonl
python final_system.py --model google/gemma-3-27b-pt --split test --out out/test.jsonl
```

`val` prints the official score, `test` writes a file for
`scripts/package_submission.py`. `python base_completion/tests.py` needs no GPU.

## Reading the scores

The metric averages F1 over all 475 rows, not over the six relations, so a
relation counts in proportion to its row count. `awardWonBy` has 10 rows and
can move the total by at most 0.02. An empty prediction against an empty gold
set scores 1.0.

## Rules

- One model, `gemma-3-27b-pt`, 27.43B parameters, measured from the weight
  shards rather than the name. Every stage calls it.
- Closed book: prompts contain the subject and training rows, nothing else.
- Nothing is fine-tuned. The only fitted values are one threshold per relation
  and the alias lists, both from the training split.
- All post-processing is non-neural string handling, which the rules allow.
