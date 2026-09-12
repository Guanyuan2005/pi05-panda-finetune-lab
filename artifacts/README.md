# Experiment artifacts

This directory contains the small, reviewable artifacts needed to audit the reported results.

- [`evaluations/`](./evaluations/README.md): immutable summaries, per-episode records, and run logs grouped by checkpoint and execution setting.
- [`videos/teacher/`](./videos/teacher/): one representative teacher demonstration for the nominal task and each recovery type.
- [`videos/checkpoint-comparison/`](./videos/checkpoint-comparison/README.md): paired 50k/100k policy videos for all improvements and regressions.
- [`logs/`](./logs/): the persistent resume log from 50k through step 102460.
- [`source/`](./source/): the pinned OpenPI commit and exact training-config patch used on the cloud instance.

Raw LeRobot datasets and Orbax checkpoints are intentionally not stored in Git. The dataset is about 18 GB locally and a single checkpoint is about 8.9 GB. Their exact configuration, validation statistics, and cloud paths are recorded in the project reports.
