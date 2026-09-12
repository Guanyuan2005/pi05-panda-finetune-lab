# Paired 50k vs 100k policy videos

All pairs use the same frozen scene seed and `smooth_alpha=1.0`. The filename records the evaluator outcome. These are precisely the 13 pairs whose binary success result changed between checkpoints.

## Normal scenes

| Seed / target | 50k | 100k | Change |
|---|---|---|---|
| 50008 / apple | [failure](./normal/seed-50008-apple/50k-failure.mp4) | [success](./normal/seed-50008-apple/100k-success.mp4) | improved |
| 50035 / box | [failure](./normal/seed-50035-box/50k-failure.mp4) | [success](./normal/seed-50035-box/100k-success.mp4) | improved |
| 50005 / orange | [success](./normal/seed-50005-orange/50k-success.mp4) | [failure](./normal/seed-50005-orange/100k-failure.mp4) | regressed |
| 50025 / orange | [failure](./normal/seed-50025-orange/50k-failure.mp4) | [success](./normal/seed-50025-orange/100k-success.mp4) | improved |
| 50029 / orange | [failure](./normal/seed-50029-orange/50k-failure.mp4) | [success](./normal/seed-50029-orange/100k-success.mp4) | improved |

## Recovery scenes

| Seed / target / recovery | 50k | 100k | Change |
|---|---|---|---|
| 50056 / apple / object slip | [failure](./recovery/seed-50056-apple-object-slip/50k-failure.mp4) | [success](./recovery/seed-50056-apple-object-slip/100k-success.mp4) | improved, boundary success at step 100 |
| 50076 / apple / object slip | [failure](./recovery/seed-50076-apple-object-slip/50k-failure.mp4) | [success](./recovery/seed-50076-apple-object-slip/100k-success.mp4) | improved |
| 50053 / orange / place offset | [failure](./recovery/seed-50053-orange-place-offset/50k-failure.mp4) | [success](./recovery/seed-50053-orange-place-offset/100k-success.mp4) | improved, boundary success at step 100 |
| 50069 / orange / place offset | [failure](./recovery/seed-50069-orange-place-offset/50k-failure.mp4) | [success](./recovery/seed-50069-orange-place-offset/100k-success.mp4) | improved |
| 50077 / orange / place offset | [failure](./recovery/seed-50077-orange-place-offset/50k-failure.mp4) | [success](./recovery/seed-50077-orange-place-offset/100k-success.mp4) | improved |
| 50085 / orange / place offset | [failure](./recovery/seed-50085-orange-place-offset/50k-failure.mp4) | [success](./recovery/seed-50085-orange-place-offset/100k-success.mp4) | improved |
| 50089 / orange / place offset | [failure](./recovery/seed-50089-orange-place-offset/50k-failure.mp4) | [success](./recovery/seed-50089-orange-place-offset/100k-success.mp4) | improved |
| 50071 / box / weak grasp | [success](./recovery/seed-50071-box-weak-grasp/50k-success.mp4) | [failure](./recovery/seed-50071-box-weak-grasp/100k-failure.mp4) | regressed |
