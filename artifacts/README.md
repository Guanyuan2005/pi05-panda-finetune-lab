# Experiment artifacts

This directory contains the small, reviewable artifacts needed to audit the reported results.

- `evaluations/`: immutable evaluator summaries and selected-scene manifests, grouped by checkpoint and execution setting.
- `videos/teacher/`: one representative teacher demonstration for the nominal task and each recovery type.
- `videos/checkpoint-comparison/`: paired 50k/100k policy videos for improvements, regressions, and remaining failures (added from the cloud export).
- `logs/`: compressed persistent training log and cloud source-diff metadata (added from the cloud export).

Raw LeRobot datasets and Orbax checkpoints are intentionally not stored in Git. The dataset is about 18 GB locally and a single checkpoint is about 8.9 GB. Their exact configuration, validation statistics, and cloud paths are recorded in the project reports.
