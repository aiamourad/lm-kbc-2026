# Instruction-tuned System

The instruction-tuned system, run through each checkpoint's **chat** interface.
It exists so the instruction tax is measured against a strong opponent rather
than a strawman. This is the "it, chat" column of the paper, defined in
Appendix C.

Official: **0.5681** with `gemma-3-27b-it`, **0.6127** with
`Mistral-Small-3.2-24B`, 0.6161 with the border rules.

## Files

| | |
|---|---|
| `run.py` | entry point: `predict`, `fit`, `score`, `sweep`, `ablate`, `validate`, `package` |
| `propose.py` | **Propose** — approach instructions, sampling and constrained decoding, set expansion for `awardWonBy` |
| `group.py` | **Group** — the scorer's normalisation, 5% tolerance clusters |
| `decode.py` | **Verification** and **Decode** — the Yes/No log-probability read, applying the calibrator, the expected-F1 argmax |
| `clients.py` | vLLM client, response cache, 32B budget check |
| `data.py` | dataset loading and splits |
| `calibrate.py` | **Calibrate** — fits the per-relation logistic regression offline from one traced run |

## Running it

```bash
python -m chat_comparator.run predict --split train --model google/gemma-3-27b-it --proposer auto
python -m chat_comparator.run fit     --checkpoint runs/train.checkpoint.jsonl
python -m chat_comparator.run predict --split val   --model google/gemma-3-27b-it --proposer auto
python -m chat_comparator.run score   --split val --predictions runs/val.jsonl
```

`--proposer auto` produced the reported scores: plain sampling everywhere, plus
set expansion for `awardWonBy`, where recall over a several-hundred-name answer
set is the bottleneck. The other choices are `sampling`, `structured` and
`expansion`.
