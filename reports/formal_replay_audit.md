# 正式数据动作回放与视觉检查

总体结果：通过

| episode | 场景 | 原记录成功 | 独立回放成功 |
| ---: | --- | --- | --- |
| 0 | normal / apple | true | true |
| 300 | approach_offset / apple | true | true |
| 301 | weak_grasp / orange | true | true |
| 302 | object_slip / blue can | true | true |
| 303 | place_offset / yellow box | true | true |

- 五条回放均未运行教师，只执行数据集 action。
- 外部相机覆盖机器人、目标物与绿色箱子；腕部相机在接近、抓取和放置阶段具有有效视野。
- 已查看每条回放每 5 秒一帧的双相机联系表；目标颜色/类别、移动过程和最终入箱结果一致。
- 量化抽样另见 `formal_sample_audit.md`：60 episodes / 180 frames，检查通过。
