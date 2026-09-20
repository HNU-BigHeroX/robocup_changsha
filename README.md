# MPE2 协同覆盖选拔赛

这是本次比赛的参赛公开版。你需要编写一个多机器人协同策略，让机器人在有限观测下覆盖移动目标，同时尽量减少碰撞。仓库提供任务环境、策略接口、公开测试配置和一份可以直接运行的学习策略示例。

完整的任务定义、评分公式和提交要求见[赛题与规程](competitionv1.0.md)。第一次使用时，可以先按下面的步骤跑通模板，再开始改自己的策略。

## 开始参赛

先向组织方确认参赛编号和仓库权限。文档用 `P017` 举例，操作时请换成自己的编号。

1. 按[获取仓库](docs/how_to_fork.md) Fork 并克隆仓库，检查并准备好本人编号分支，再复制模板建立本人目录。
2. 按[使用指南](docs/how_to_use.md)安装 Python 3.12 和评测依赖。
3. 跑一次提交预检和公开测试，确认模板在你的机器上能够正常运行。
4. 在 `participant/P017/` 内开发，记录实验，按[参与指南](CONTRIBUTING.md)准备最终提交。

使用 AI 辅助开发时，请让助手先读取 [AGENTS.md](AGENTS.md)。其中约定了 HUNer 的工作范围和开发流程。

## 环境与运行约定

- Python 必须为 3.12，项目声明的支持范围是 `>=3.12,<3.13`。
- 安装、预检和评测命令都应在仓库根目录执行。
- 评测环境与训练环境分开创建；最终推理不能依赖只安装在训练环境中的框架。
- `participant/P017/` 表示本人的提交目录，文档中的 `P017` 必须统一替换为组织方分配的编号。
- `outputs/P017/`、虚拟环境和缓存只用于本地运行，不属于参赛提交。

安装完成后可以先做最小自检：

```sh
python --version
python -m pip check
python -c "from coverage_bench import get_protocol_spec; print(get_protocol_spec())"
```

第一条命令应显示 Python 3.12，第二条命令应报告依赖关系正常，第三条命令应输出当前协议规格。若这里失败，先修复环境，不要直接进入训练或评测。

## 先跑一次公开测试

以下命令在参赛仓库根目录执行，`python` 应指向已经配置好的评测环境。开始前，请确认本人目录已经建立，且 `submission.yaml` 中的 `participant_id` 与目录名一致。

```sh
python scripts/check_submission.py --submission participant/P017 --output outputs/P017/check
python scripts/evaluate_one.py --submission participant/P017 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output outputs/P017/eval
```

先查看 `outputs/P017/eval/result.json` 中的 `status`；成功时，`performance_score` 就是这次公开测试的性能分。逐回合数据在同目录的 `episodes.csv`。

评测脚本会检查当前平台是否提供 `setitimer`、`SIGALRM` 和 `ITIMER_REAL`：不支持时不执行阶段超时限制，并将结果标记为 `local_preview`；支持时会启用阶段超时。即使本地结果的 `provenance` 为 `official_container`，也只表示采用了带阶段超时的执行路径，不代表已经获得组织方核验。不同平台的数值差异也可能影响策略表现，正式成绩仍以组织方统一核验为准。

一次运行是否有效，建议按下面的顺序判断：

1. 预检命令退出码为 0，且 `audit-report.json` 中没有 `rejections`。
2. 评测命令退出码为 0，且 `result.json` 中的 `status` 为 `ok`。
3. 确认 `participant_id`、`protocol_version`、`task_version` 和 `suite_id` 与本次运行一致。
4. 再读取 `performance_score` 和分组指标；失败运行不会产生有效性能分。
5. 查看 `episodes.csv`，确认 8 个公开测试回合均成功完成，而不是只看总分。

## 推荐的开发循环

每次实验只改变一个主要因素，能够更容易判断改动是否有效：

1. 在 `LOG.md` 写下问题、假设和计划改变的内容。
2. 在本人目录内修改策略、训练脚本或个人配置。
3. 训练或导出模型；替换模型后同步更新 `submission.yaml` 中的摘要与字节数。
4. 运行提交预检，再运行固定公开套件。
5. 将代码版本、训练配置、种子、分组指标和结论写入 `experiments.csv` 与 `LOG.md`。
6. 保留效果明确的改动；失败实验也记录原因，避免之后重复尝试。

公开测试适合验证接口和比较方案，但不应被反复用作唯一训练目标。最终报告应能说明策略原理、复现方法、关键对照、失败场景和资源使用情况。

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

## 提交前自查

- 当前位于本人编号分支或由其创建的功能分支。
- `git status --short` 和暂存区文件列表中没有官方代码、配置、模板或其他参赛者目录。
- `submission.yaml` 的编号、方法类型、训练命令、产物路径、SHA-256 和字节数与实际文件一致。
- 推理只依赖 `requirements-infer.lock` 中允许的依赖，并能在 CPU、断网环境中运行。
- `reset` 会清空上一回合状态，`act` 始终返回合法的 `(2,)` `float32` 动作。
- `LOG.md`、`experiments.csv`、`REPORT.md` 和 `THIRD_PARTY.md` 已按真实情况更新。
- 提交预检和公开测试均已重新运行，结果目录与旧实验分开保存。
- 没有暂存虚拟环境、缓存、训练检查点或 `outputs/` 下的本地结果。

## 文档

- [参与指南](CONTRIBUTING.md)：开发记录、报分和最终提交。
- [获取仓库](docs/how_to_fork.md)：Fork、克隆、准备分支、复制模板。
- [使用指南](docs/how_to_use.md)：环境安装、接口、模型和本地测试。
- [Git 工作流程](docs/git.md)：编号分支、功能分支、同步更新和标签。
- [提交消息](docs/cz.md)：怎样写清楚每次提交做了什么。
- [赛题与规程](competitionv1.0.md)：任务定义和比赛规则。
