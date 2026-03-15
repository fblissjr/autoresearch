---
name: compare
description: Compare training runs from data/run_*.json files
argument-hint: "[count]"
disable-model-invocation: true
---

# Compare Runs

Read `data/run_*.json` files and produce a comparison table.

## Arguments

- `count` (optional): Number of most recent runs to compare. Default: all available.

## Steps

### 1. Find run files

```bash
ls -t data/run_*.json
```

If a count argument was provided, take only the most recent N files.

### 2. Extract key metrics from each run

For each `data/run_*.json`, extract:
- `timestamp`
- `result.val_bpb`
- `training.total_steps`
- `training.avg_tok_sec`
- `training.peak_memory_mb` (overall peak)
- `training.actual_seconds` (training time)
- `training.total_seconds` (training + eval)
- `model.depth`
- `model.n_embd`
- `training.batch_size`

If `training_peak_mb` exists (v0.5.3+), include it separately from overall peak.

### 3. Output comparison table

Format as a markdown table sorted by timestamp (newest first):

```
| Run | val_bpb | Steps | Avg tok/s | Train Peak | Overall Peak | Train (s) | Total (s) |
|-----|---------|-------|-----------|------------|--------------|-----------|-----------|
| <timestamp> | X.XXX | XXX | XX,XXX | XX.X GB | XX.X GB | XXX.X | XXX.X |
```

### 4. Highlight best

Mark the best val_bpb with `**bold**`. If the most recent run improved over the previous best, note the delta.

### 5. Throughput analysis

If there are 2+ runs, compute:
- Throughput trend (improving, stable, regressing)
- Memory trend (growing, stable, shrinking)
- Steps trend (more steps = more training in the same budget = good)

### 6. Step timing analysis (optional)

If `step_timings` exists in the JSON, compute:
- Average step time across all steps
- Whether throughput was flat or had regressions (compare first 20% vs last 20% of steps)
- Memory stability (compare first and last memory samples)
