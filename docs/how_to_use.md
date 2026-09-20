# 使用指南

本文中的命令都在参赛仓库根目录执行，示例编号为 `P017`。开始前，先按[获取仓库](how_to_fork.md)确认编号、准备分支并创建本人目录。已有目录时不用重复复制模板。

## 配置评测环境

使用 Python 3.12。建议为评测单独建一个虚拟环境，安装仓库提供的锁定依赖。

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-official.lock
.venv/Scripts/python.exe -m pip install --no-deps -e .
```

Linux / macOS：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-official.lock
.venv/bin/python -m pip install --no-deps -e .
```

下文用 `python` 简写评测环境的解释器。你可以激活环境后执行，也可以直接替换成 `.venv/Scripts/python.exe` 或 `.venv/bin/python`。先安装锁定依赖，再以 `--no-deps` 安装本仓库，避免安装项目时重新解析运行依赖。

虚拟环境、Python 缓存和 `outputs/` 下的测试结果只保留在本地。提交时只暂存本人编号目录中的参赛材料，不要使用 `git add .` 把它们一起加入。

### 安装后自检

激活评测环境后，从仓库根目录执行：

```sh
python --version
python -m pip check
python -c "import numpy, gymnasium, pettingzoo, mpe2; print('runtime imports: ok')"
python -c "from coverage_bench import get_protocol_spec; print(get_protocol_spec())"
```

应确认：

- Python 版本为 3.12；
- `pip check` 没有报告缺失依赖或版本冲突；
- 运行依赖都能导入；
- 协议规格中的 `protocol_version` 和 `task_version` 能正常读取。

如果使用多个虚拟环境，还可以运行 `python -c "import sys; print(sys.executable)"`，确认当前命令没有误用训练环境或系统 Python。

## 跑通模板

`participant/_template/` 是一份 PPO 学习策略示例，已经带有导出的 `artifacts/policy.npz`。复制并改好编号后，可以直接评测，不必先训练。

评测程序从 `entry.py` 调用 `build_policy(context)`。返回的策略对象需要实现：

| 方法 | 作用 |
| --- | --- |
| `reset(episode_context)` | 初始化新回合，清空上回合的内部状态 |
| `act(observation)` | 根据单个机器人的局部观测返回动作 |
| `close()` | 释放策略使用的资源 |

请结合模板的[策略实现](../participant/_template/entry.py)和[协议类型](../coverage_bench/protocol.py)查看完整签名。动作必须是形状为 `(2,)` 的 `float32` 数组，两个分量都在 `[-1, 1]` 内。

模型、配置和材料要求见[赛题与规程](../MPE2-协同覆盖选拔赛-赛题与规程.md)第 16 节。模板中的报告、日志和实验数据用于展示格式，提交时要换成自己的实际记录。

### 运行前确认

复制模板后，至少检查以下内容：

| 检查项 | 正确状态 |
| --- | --- |
| 目录名 | 与组织方分配的 `Pxxx` 编号一致 |
| `participant_id` | 与目录名完全一致 |
| `entrypoint` | 保持为 `entry:build_policy` |
| `artifacts/` | 包含正式推理实际读取的全部文件 |
| `checkpoint_manifest` | 每个登记文件的路径、SHA-256 和字节数都与磁盘文件一致 |
| `requirements-infer.lock` | 只包含正式推理所需依赖 |
| 记录材料 | 不再包含模板虚构的实验结果或结论 |

建议为每次实验使用独立输出目录，例如 `outputs/P017/eval-001`、`outputs/P017/eval-002`。评测程序会写入指定目录；混用目录会让不同版本的结果难以对应。

## 提交预检与公开测试

先检查目录结构、清单、模型摘要和静态审核项：

```sh
python scripts/check_submission.py --submission participant/P017 --output outputs/P017/check
```

通过后再运行公开测试：

```sh
python scripts/evaluate_one.py --submission participant/P017 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output outputs/P017/eval
```

输出文件都在指定的 `--output` 目录中：

| 文件 | 查看内容 |
| --- | --- |
| `result.json` | 运行状态、分组指标、`performance_score` 和版本信息 |
| `episodes.csv` | 每个回合的结果 |
| `timings.npz` | 耗时数据 |

预检输出目录中的 `audit-report.json` 还包含以下信息：

| 字段 | 含义 |
| --- | --- |
| `rejections` | 硬拒绝项；非空时不能进入有效评测 |
| `scan_hits` | 需要人工确认的扫描命中；命中本身不等于硬拒绝 |
| `digest` | 编号、协议版本、代码树和依赖锁摘要等溯源信息 |

两个脚本的退出码可以用于自动化判断：

| 命令 | 退出码 | 含义 |
| --- | ---: | --- |
| `check_submission.py` | 0 | 结构校验与静态审核通过 |
| `check_submission.py` | 2 | 输入、清单、摘要或静态审核未通过 |
| `evaluate_one.py` | 0 | 全部评测成功，结果状态为 `ok` |
| `evaluate_one.py` | 2 | 提交、套件或种子配置错误 |
| `evaluate_one.py` | 3 | 加载、超时、协议或运行时错误 |
| `evaluate_one.py` | 4 | 评测执行出现其他致命错误 |

先确认 `result.json` 中的 `status` 为 `ok`，再比较成绩；失败结果不能作为有效分数。当前公开套件有 4 个场景，每个场景重复 2 次，共 8 个回合。评分定义见规程第 17 节，配置见 [scoring-v1.yaml](../configs/scoring-v1.yaml)。

读取结果时不要只看 `performance_score`。还应核对：

- `provenance`：区分正式容器结果与 `local_preview`；
- `working_tree_dirty` 和 `source_commit`：帮助定位结果对应的代码状态；
- `group_metrics`：分别观察基础组和协作组表现；
- `errors`：失败时查看错误阶段、错误码和定位信息；
- `episodes.csv` 的 `status`、`steps_completed`、覆盖率、碰撞率和耗时字段。

Windows 和 macOS 不支持这里使用的阶段超时机制，脚本会进入 `local_preview` 模式，初始化、单步和单回合的超时限制不生效。Linux 上的本地运行也不能替代组织方核验。不同平台可能产生数值差异，正式成绩以组织方统一环境中的结果为准。

## 使用训练接口

仓库提供以下接口，适合在本人目录中编写训练或测试脚本：

| 接口 | 用途 |
| --- | --- |
| `coverage_bench.load_task_config(path)` | 读取任务配置，返回 `TaskConfig` |
| `coverage_bench.make_training_env(config)` | 创建 PettingZoo 并行环境 |
| `env.reset(seed=...)` | 开始一个回合，返回各机器人的观测与信息 |
| `env.step(actions)` | 同时执行各机器人的动作 |
| `env.state()` | 获取训练专用的全局状态 |
| `coverage_bench.get_protocol_spec()` | 读取观测、动作和容量规格 |
| `coverage_bench.spaces.flatten_observation(obs, spec)` | 将单机器人观测展平为当前协议下的 104 维向量 |

下面的示例执行一步零动作。将它保存到本人目录中的 Python 文件后，用评测环境运行即可：

```python
from pathlib import Path

import numpy as np

from coverage_bench import load_task_config, make_training_env

config = load_task_config(Path("configs/task-v1.yaml"))
env = make_training_env(config)
try:
    observations, infos = env.reset(seed=0)
    actions = {
        agent: np.zeros(2, dtype=np.float32)
        for agent in env.agents
    }
    observations, rewards, terminated, truncated, infos = env.step(actions)
finally:
    env.close()
```

`env.state()` 只用于训练，正式推理不能访问它。策略收到的是单机器人的局部观测，字段、掩码和归一化约定见规程第 12、13、15 节。训练时也请使用仓库提供的展平函数，保持训练与推理的特征顺序一致。

### 训练代码的最小检查

训练脚本应明确控制随机种子、训练步数、输出目录和配置来源。开始长时间训练前，先用很少的步数完成一次冒烟测试，确认能够：

1. 创建环境并完成 `reset`；
2. 执行若干次 `step`；
3. 保存训练产物和元信息；
4. 正常关闭环境；
5. 在新的 Python 进程中重新加载保存结果。

冒烟测试只证明流程能够运行，不代表策略有效。正式比较时，应固定评测套件和种子安排，并记录代码版本与模型摘要。

## 配置训练环境（可选）

需要训练学习策略时，再建一套训练环境。它包含 PyTorch、Stable-Baselines3 和 SuperSuit，不用于最终提交的推理验证。

Windows PowerShell：

```powershell
py -3.12 -m venv .venv-train
.venv-train/Scripts/python.exe -m pip install -r requirements-train.lock
.venv-train/Scripts/python.exe -m pip install --no-deps -e .
.venv-train/Scripts/python.exe participant/P017/train.py --total-steps 1000000 --out outputs/P017/training
```

Linux / macOS：

```sh
python3.12 -m venv .venv-train
.venv-train/bin/python -m pip install -r requirements-train.lock
.venv-train/bin/python -m pip install --no-deps -e .
.venv-train/bin/python participant/P017/train.py --total-steps 1000000 --out outputs/P017/training
```

模板训练脚本会保存训练模型、归一化统计量、训练曲线和元信息。它不会自动把新模型导出到 `artifacts/policy.npz`，也不会替你更新提交清单。公开包没有附带独立的导出工具；重新训练后，需要在本人目录内编写导出代码，按推理实现需要的格式保存权重和归一化统计量。

## 模型与提交清单

评测环境没有 PyTorch 或 Stable-Baselines3。模型需要能在断网、CPU 环境中加载和推理。模板采用 `.npz` 加 NumPy 的方式；提交源码中禁止 `pickle.load`、`torch.load` 等不安全反序列化调用。

正式推理需要的文件放在本人目录的 `artifacts/` 中，清单使用本人目录内的相对路径。不要用外部下载地址代替模型，也不要让路径跳出本人目录。训练检查点留在本地，提交用于推理的安全格式产物。

每次替换模型后，更新 `submission.yaml` 中对应的 `checkpoint_manifest` 项：`path`、`sha256` 和 `size_bytes`。可以用下面的命令读取文件大小和 SHA-256：

```sh
python -c "from pathlib import Path; import hashlib; p = Path('participant/P017/artifacts/policy.npz'); print('size_bytes:', p.stat().st_size); print('sha256:', hashlib.sha256(p.read_bytes()).hexdigest())"
```

同步更新训练命令、个人配置路径、推理依赖和报告，再重新运行预检。导出模型后，还要比较训练模型与推理模型在同一批观测上的动作；预检通过并不能证明两者一致。

建议按以下顺序替换模型：

1. 在训练环境中完成导出，并保存导出脚本使用的配置和随机种子。
2. 在评测环境中单独加载导出文件，确认没有隐式依赖 PyTorch、Stable-Baselines3 或训练目录。
3. 对固定的一批观测比较训练模型与推理实现的动作，记录最大绝对误差和允许误差。
4. 将正式产物复制到本人目录的 `artifacts/`，删除不参与推理的临时检查点。
5. 更新 `checkpoint_manifest`、训练命令、配置路径和实验记录。
6. 重新执行提交预检和完整公开测试。

## 常见问题排查

### 找不到 `submission.yaml`

确认 `--submission` 指向编号目录，而不是仓库根目录或 `entry.py`。目录中应直接包含 `submission.yaml`。

### 目录名与参赛编号不匹配

编号目录名和 `submission.yaml` 中的 `participant_id` 必须完全相同，并符合 `P` 加三位数字的格式。不要只修改其中一处。

### 模型摘要或字节数不匹配

每次重新导出、复制或修改产物后都要重新计算 SHA-256 和 `size_bytes`。确认清单记录的是本人目录中的正式文件，而不是训练输出目录中的同名文件。

### `entry.py` 可以导入，但评测仍失败

如果输出文件已经生成，依次查看 `result.json` 的 `status` 和 `errors`，再在 `episodes.csv` 中定位失败的场景、回合、步骤和 `error_code`。如果没有生成结果文件，先读取命令行错误信息。常见原因包括：返回动作形状或类型错误、动作包含非有限数值、模型文件路径依赖当前工作目录、`reset` 没有清空状态，或推理超过时间限制。

### 本地成绩和正式成绩不同

先比较 `provenance`、平台、依赖锁、代码提交、工作区状态、模型摘要、套件与种子安排。Windows 和 macOS 的 `local_preview` 不执行阶段超时限制；即使输入相同，平台数值差异也可能改变边界行为。正式成绩始终以组织方统一环境核验为准。

### 预检出现 `scan_hits`

`scan_hits` 用于提示人工审核，不会单独造成硬拒绝。检查命中的文件、行号和符号，确认其用途能够在报告中解释。`rejections` 才表示必须修复的硬拒绝项。

## 完成一次实验的记录清单

- 代码提交或可定位的代码版本；
- 训练配置、训练种子、训练步数和运行环境；
- 模型路径、字节数和 SHA-256；
- 公开套件、种子安排和结果目录；
- 各分组指标、总分、运行状态和耗时；
- 与对照方案的唯一主要差异；
- 结论、失败现象和下一步。

只有实际运行过的结果才能写入实验记录。失败运行同样应保留状态和原因，不要把预计结果或模板示例当作实测结论。

## 参考资料

遇到 API 用法问题，可以查这些项目的官方文档和源码。安装版本仍以本仓库的依赖锁为准。

- MPE2：[文档](https://mpe2.farama.org/)、[源码](https://github.com/Farama-Foundation/MPE2)。
- PettingZoo：[并行环境 API](https://pettingzoo.farama.org/api/parallel/)、[源码](https://github.com/Farama-Foundation/PettingZoo)。
- Stable-Baselines3：[文档](https://stable-baselines3.readthedocs.io/en/master/)、[源码](https://github.com/DLR-RM/stable-baselines3)。

这些资料用于理解依赖库，比赛中的任务、观测和动作定义以本仓库为准。引用第三方代码或模型时，在本人目录的 `THIRD_PARTY.md` 中记录来源和许可证。
