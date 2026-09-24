# P717 策略研究报告

## 当前状态

当前版本完成了参赛目录初始化、评测环境配置和官方模板基线复现，尚未实现个人策略，也未训练个人模型。模板自带模型仅用于验证提交与评测流程。

## E001：官方模板基线

- 日期：2026-09-24
- 代码提交：`7a9cc0722a61b8345ba53b2d5fe5f32b5053993f`
- Python：3.12.10
- 测试套件：`public-suite-v1`
- 运行标记：`local_preview`
- 有效回合：8/8
- `performance_score`：66.6666666666667
- basic `mean_j`：0.0
- cooperation `mean_j`：0.13333333333333333
- 两组平均碰撞率：0.0

该结果来自 Windows 本地预览，未执行官方 Linux 环境中的阶段超时限制，不是组织方核验成绩。

## 复现命令

在仓库根目录使用 Python 3.12 评测环境执行：

```powershell
.venv\Scripts\python.exe scripts/check_submission.py --submission participant/P717 --output outputs/P717/check-template
.venv\Scripts\python.exe scripts/evaluate_one.py --submission participant/P717 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output outputs/P717/eval-template
```

## 后续计划

1. 建立基于目标追踪、确定性分工和机器人间避碰的规则策略。
2. 将规则策略与官方模板分别在相同公开套件上比较。
3. 在规则基线稳定后，再评估强化学习或规则与学习混合方案。

## 已知局限

- 当前策略仍是官方模板，不代表最终方案。
- 尚未进行个人训练、重复训练种子实验或训练模型导出一致性验证。
- 当前只有 Windows `local_preview` 结果，最终成绩以组织方统一环境核验为准。
