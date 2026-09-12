# h50 单 episode 过拟合结果

- 配置：`action_horizon=50`、batch=1
- 数据：1 个完整 Panda episode
- 训练：5000 steps，loss 约从 0.105 降至 0.003–0.005
- 同场景闭环（每次执行 1 步）：0/1
- 同场景闭环（每次执行 5 步）：0/1

视频：

- `results/h50_overfit/episode_000_exec1.mp4`
- `results/h50_overfit/episode_000_exec5.mp4`

结论：h50 能在显存上运行，也能把训练 loss 拟合到很低，但单回合闭环仍失败。因此当前主要问题不是 action horizon 太短，也不是单纯训练步数不足；需要继续检查模型输出与执行控制之间的语义/实现，或更换训练目标与闭环策略。

## 追加：执行 10 步

同一 checkpoint、同一 overfit 场景，每次观测后执行预测序列前 10 步：`success=true`，成功率 `1/1`，第 29 个控制周期完成，平均推理耗时约 825 ms（首轮编译影响）。成功视频为 `results/h50_overfit/episode_000_exec10_success.mp4`。

这说明模型确实学到了抓取轨迹；执行 1 步会反复重规划，动作推进太短，执行 5 步已有明显抓取动作但未完成，执行 10 步完成抓取。

## 视频时长修正

原评估器每个 chunk 只写 1 帧，因此执行 10 步时视频被压缩约 10 倍；这不代表物理仿真加速。已修正为每个 100 ms 动作写一帧，并重新生成成功视频：`results/h50_overfit/episode_000_exec10_success_fixed.mp4`。修正后仍成功 `1/1`。

## 动作平滑

加入 EMA 动作平滑（alpha=0.3）和每个 100ms 控制周期内的目标插值后，同一 h50 过拟合场景仍成功 `1/1`，视频为 `results/h50_overfit/episode_000_exec10_smooth03.mp4`。
