# MPE2 协同覆盖选拔赛

这是本次比赛的参赛公开版。你需要编写一个多机器人协同策略，让机器人在有限观测下覆盖移动目标，同时尽量减少碰撞。仓库提供任务环境、策略接口、公开测试配置和一份可以直接运行的学习策略示例。

完整的任务定义、评分公式和提交要求见[赛题与规程](MPE2-协同覆盖选拔赛-赛题与规程.md)。第一次使用时，可以先按下面的步骤跑通模板，再开始改自己的策略。

## 开始参赛

先向组织方确认参赛编号和仓库权限。文档用 `P017` 举例，操作时请换成自己的编号。

1. 按[获取仓库](docs/how_to_fork.md)克隆仓库，检查并准备好本人编号分支，再复制模板建立本人目录。
2. 按[使用指南](docs/how_to_use.md)安装 Python 3.12 和评测依赖。
3. 跑一次提交预检和公开测试，确认模板在你的机器上能够正常运行。
4. 在 `participant/P017/` 内开发，记录实验，按[参与指南](CONTRIBUTING.md)准备最终提交。

使用 AI 辅助开发时，请让助手先读取 [AGENTS.md](AGENTS.md)。其中约定了 HUNer 的工作范围和开发流程。

## 先跑一次公开测试

以下命令在参赛仓库根目录执行，`python` 应指向已经配置好的评测环境。开始前，请确认本人目录已经建立，且 `submission.yaml` 中的 `participant_id` 与目录名一致。

```sh
python scripts/check_submission.py --submission participant/P017 --output outputs/P017/check
python scripts/evaluate_one.py --submission participant/P017 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output outputs/P017/eval
```

先查看 `outputs/P017/eval/result.json` 中的 `status`；成功时，`performance_score` 就是这次公开测试的性能分。逐回合数据在同目录的 `episodes.csv`。

Windows 和 macOS 上的运行属于本地预览，不执行阶段超时限制，结果标记为 `local_preview`。不同平台的数值差异也可能影响策略表现。本地测试用于检查运行情况、比较自己的方案，正式成绩以组织方统一核验为准。

## 仓库里有什么

| 路径 | 用途 |
| --- | --- |
| `participant/_template/` | PPO 学习策略示例，包含训练代码、NumPy 推理代码、模型和材料样例 |
| `participant/P017/` | 按自己的编号创建，存放策略、模型和实验记录 |
| `coverage_bench/` | 官方任务环境、观测与动作协议、评测实现 |
| `configs/` | 公开测试场景、种子和评分配置 |
| `scripts/check_submission.py` | 检查提交结构、模型摘要和静态审核项 |
| `scripts/evaluate_one.py` | 运行一份提交，生成本地测试结果 |

参赛期间只能修改本人编号目录内的提交文件。官方代码、配置、模板和其他参赛者的目录由各自维护者负责；虚拟环境、缓存和本地测试输出不要放进提交。

## 开发前需要知道

策略入口是 `entry.py` 中的 `build_policy(context)`，返回的对象需要实现 `reset`、`act` 和 `close`。每个机器人的动作是形状为 `(2,)` 的 `float32` 数组，两个分量都在 `[-1, 1]` 内。接口和采样示例见[使用指南](docs/how_to_use.md)。

训练和评测使用两套依赖。训练环境可以使用 PyTorch、Stable-Baselines3；评测环境没有这些框架。模板通过 `.npz` 保存权重，用 NumPy 完成推理。训练后需要自行导出模型，并更新提交清单中的文件摘要，具体要求见[模型与提交清单](docs/how_to_use.md#模型与提交清单)。

## 文档

- [参与指南](CONTRIBUTING.md)：开发记录、报分和最终提交。
- [获取仓库](docs/how_to_fork.md)：克隆、准备分支、复制模板。
- [使用指南](docs/how_to_use.md)：环境安装、接口、模型和本地测试。
- [Git 工作流程](docs/git.md)：编号分支、功能分支、同步更新和标签。
- [提交消息](docs/cz.md)：怎样写清楚每次提交做了什么。
- [赛题与规程](MPE2-协同覆盖选拔赛-赛题与规程.md)：任务定义和比赛规则。
