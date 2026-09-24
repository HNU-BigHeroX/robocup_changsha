# P717 策略研究报告

## 当前状态

P717 已完成参赛目录、Python 3.12 评测环境和模板流程验证，并实现无学习权重的规则策略。E001 保留为官方模板的本地基线；规则策略的成绩只有在干净工作树完成预检和公开评测后才写入。

## 方法

正式策略是不加载学习权重的无状态规则策略。每个机器人只读取本地观测中同时满足 `exists` 与 `visible` 的槽位，按自身到目标的距离稳定排序。对每个候选目标，若某个可见同伴距离该目标更近，则当前机器人让位并尝试下一目标；距离精确相等时，仅由较小 `agent_index` 获胜以打破对称。选中目标后按两个坐标分量分别输出 `-1` 或 `+1`，仅在目标相对向量恰为零或无可用目标时输出零动作。

该实现支持协议固定容量 8 和空槽掩码，不在策略中读取案例编号、评测种子、隐藏配置或公开套件分数。

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

在仓库根目录使用仓库外的 Python 3.12 评测环境执行；每次为 `$P717CheckOutput` 和 `$P717EvalOutput` 指定新的仓库外目录：

```powershell
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' scripts/check_submission.py --submission participant/P717 --output $P717CheckOutput
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' scripts/evaluate_one.py --submission participant/P717 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output $P717EvalOutput
```

## 已知局限

- 当前规则策略不保留跨步记忆；目标不可见或所有目标均让位时输出零动作。
- Windows `local_preview` 不启用官方阶段超时，正式成绩以组织方统一核验为准。
- 当前设计依据公开任务协议和公开对照实验，不把公开套件结果硬编码进策略。
