---
name: data
description: Launch an autonomous data experiment loop (edits prepare.py + data_sources.py). Usage -- /experiment:data <run-tag> [dataset]
argument-hint: "<run-tag> [dataset]"
disable-model-invocation: true
---

# Data Experiment Loop

Parse arguments from `$ARGUMENTS`:
- First token: **run tag** (required). If missing, ask the user for one.
- Second token: **dataset** (optional, default `climbmix`).

Then:

1. Read `program_data.md` from the project root. This is the full experiment loop specification -- follow it exactly.
2. During **Setup** (step 1 of program_data.md), use the run tag and dataset parsed above.
3. Do not ask for confirmation -- start the loop immediately after setup completes.
