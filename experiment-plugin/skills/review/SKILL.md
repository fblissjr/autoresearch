---
name: review
description: Pre-flight review of train.py changes before running an experiment. Catches issues that would waste a 5-minute training run.
argument-hint: ""
disable-model-invocation: true
---

# Pre-flight Experiment Review

Run the `experiment-reviewer` agent to check the current train.py changes before committing to a 5-minute training run.

1. Check `git diff train.py` -- if there are no changes, report "nothing to review" and stop.
2. Follow the experiment-reviewer agent checklist: import safety, test suite, constraint violations, MLX pitfalls, memory concerns, performance red flags.
3. Output the structured Pre-flight Check report.
