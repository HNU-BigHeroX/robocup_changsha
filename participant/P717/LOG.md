# 实验日志

## E001：官方模板本地基线（2026-09-24）

- 目的：确认 P717 目录、Python 3.12 评测环境、提交预检和公开评测流程能够完整运行。
- 代码版本：`7a9cc0722a61b8345ba53b2d5fe5f32b5053993f`。
- 策略与模型：未训练个人模型；直接使用 `participant/_template/` 随附的 NumPy 推理策略和 `artifacts/policy.npz`。
- 运行环境：Windows，Python 3.12.10；由于平台不支持官方阶段超时机制，本次结果标记为 `local_preview`。
- 提交预检：退出码为 0，`rejections` 为空；模板 `train.py` 中的 `torch` 与 `torch.no_grad` 产生两条人工审核提示，不属于硬拒绝项。
- 公开测试：`public-suite-v1`，8/8 回合状态为 `ok`；`performance_score=66.6666666666667`。
- 分组结果：basic `mean_j=0.0`；cooperation `mean_j=0.13333333333333333`；两组平均碰撞率均为 0。
- 结论：本次实验只证明模板和本机流程可运行，不代表个人算法效果，也不代表组织方正式核验成绩。下一步建立可解释的规则策略基线。
