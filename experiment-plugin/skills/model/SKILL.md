---
name: model
description: Launch an autonomous model experiment loop (edits train.py). Usage -- /experiment:model <run-tag> [dataset]
argument-hint: "<run-tag> [dataset]"
disable-model-invocation: true
---

# Model Experiment Loop

Parse arguments from `$ARGUMENTS`:
- First token: **run tag** (required). If missing, ask the user for one.
- Second token: **dataset** (optional, default `climbmix`).

Then:

1. Read `program.md` from the project root. This is the full experiment loop specification -- follow it exactly.
2. During **Setup** (step 1 of program.md), use the run tag and dataset parsed above.
3. Do not ask for confirmation -- start the loop immediately after setup completes.
