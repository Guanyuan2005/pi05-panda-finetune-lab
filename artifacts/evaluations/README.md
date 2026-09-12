# Evaluation results

These are the evaluator outputs copied from the frozen cloud runs. Each run keeps its aggregate `summary.json`, per-episode JSON records, and console `run.log` where available.

> `recovery` in these paths is a historical metadata split. The current evaluator did not inject the named perturbation during policy rollout, so these numbers must not be interpreted as causal recovery rates.

| Checkpoint | Split / setting | Result |
|---|---|---|
| 50k | normal, alpha 0.2/0.3/0.5 quick matrix | [`normal-smoothed-quick/`](./50k/normal-smoothed-quick/) |
| 50k | normal, alpha 1.0, 40 episodes | [`summary.json`](./50k/normal-alpha1/summary.json) |
| 50k | recovery, alpha 0.2, 40 episodes | [`summary.json`](./50k/recovery-alpha02/summary.json) |
| 50k | recovery, alpha 1.0, 40 episodes | [`summary.json`](./50k/recovery-alpha1/summary.json) |
| 100k | normal, alpha 0.2, 40 episodes | [`summary.json`](./100k/normal/alpha_0.2/summary.json) |
| 100k | normal, alpha 1.0, 40 episodes | [`summary.json`](./100k/normal/alpha_1.0/summary.json) |
| 100k | recovery, alpha 0.2, 40 episodes | [`summary.json`](./100k/recovery/alpha_0.2/summary.json) |
| 100k | recovery, alpha 1.0, 40 episodes | [`summary.json`](./100k/recovery/alpha_1.0/summary.json) |

The headline comparison and paired interpretation are in [`reports/h50_formal_50k_vs_100k_report.md`](../../reports/h50_formal_50k_vs_100k_report.md).
