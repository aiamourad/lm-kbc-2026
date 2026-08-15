# Each table, and the command that produces it

Serve the checkpoint first. `--model` takes either a Hugging Face id or a
local path to the weights. Paths below assume the task's dataset
([`lm-kbc/dataset2026`](https://github.com/lm-kbc/dataset2026)) is checked out
as `dataset2026-main/` beside this directory.

Nothing here reads test gold; it is not published. Thresholds, exemplars, alias
lists and the sampling temperature are fixed on train; validation only checks.

## Final system — Table 3, 0.6836

```bash
cd base_completion
python final_system.py --model google/gemma-3-27b-pt --split val  --out out/val.jsonl
python final_system.py --model google/gemma-3-27b-pt --split test --out out/test.jsonl
python ../scripts/package_submission.py -p out/test.jsonl -g ../dataset2026-main/data/test.jsonl -o submission.zip
```

Every relation at its own budget and temperature, then the `awardWonBy` rescue,
the `personHasCityOfDeath` existence check, and the alias deduplication. Expect
val 0.669, test 0.6836.

## Knowledge probe — Table 1

```bash
python knowledge_probe.py numeric --gold ../dataset2026-main/data/val.jsonl --relation hasArea --model google/gemma-3-27b-pt
python knowledge_probe.py sets    --gold ../dataset2026-main/data/val.jsonl --relation awardWonBy --model google/gemma-3-27b-pt
```

100 validation rows, several question forms, 8–12 samples each. Reports *voted*
(best form) and *ceiling* (answer present in any sample). About one GPU-hour per
checkpoint; run once per row of Table 1.

## Instruction tax — Tables 2 and 5

The final-system command, once per checkpoint, changing only `--model`:

```bash
python final_system.py --model google/gemma-3-27b-pt --split test --out out/pt.jsonl --fit
python final_system.py --model google/gemma-3-27b-it --split test --out out/it.jsonl --fit
```

`--fit` matters: the thresholds in `fitted-supports.json` were fitted for
`gemma-3-27b-pt` and do not transfer. Everything else, including the `hasArea`
temperature, is held fixed across runs — which is what makes it a comparison.

## Instruction-tuned system — Appendix C

```bash
cd instruction_tuned
python run.py predict --model google/gemma-3-27b-it --split test --out out/chat-test.jsonl
```

The "it, chat" column of Table 2, official 0.5681. `run.py fit` fits the
calibrator; `run.py sweep` grid-searches the decoder settings.

## Tests

```bash
cd base_completion && python tests.py
```

Number parsing and units, tolerance clustering, recurrence and abstention,
prompt construction, alias deduplication. No GPU.
