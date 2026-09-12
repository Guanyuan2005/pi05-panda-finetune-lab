# 正式数据抽样审计

总体结果：通过

- episodes checked: 60
- frames checked: 180（每条首/中/末三帧、双相机）
- recovery counts: `{'none': 20, 'approach_offset': 10, 'weak_grasp': 10, 'object_slip': 10, 'place_offset': 10}`
- target counts: `{'apple': 14, 'orange': 13, 'box': 18, 'can': 15}`
- 检查项：图像 shape/range/方差/时序变化、task 与目标物标签一致、恢复事件与 retry 完整、最终成功
