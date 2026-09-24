# P717 225 分规则基线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 P717 从官方 PPO 模板替换为仅依赖局部观测的“最近可见目标 + 按距离让位 + 角点饱和”规则策略，并在干净工作树上复现公开套件 `225.00`、零碰撞和 8/8 回合成功。

**Architecture:** `entry.py` 保持单文件、无状态 NumPy 推理：先用 `exists & visible` 掩码筛选固定容量 8 的目标与同伴，再按自身距离稳定排序目标；若可见同伴离同一目标更近，则尝试下一个目标，精确同距时只用较小机器人编号打破对称。提交改为 `rule`，不加载学习权重；`artifacts/model_card.json` 仅承担必需产物目录、可审计说明和清单登记作用。

**Tech Stack:** Python 3.12、NumPy、pytest、YAML、仓库自带 `check_submission.py` 与 `evaluate_one.py`。

---

## 文件结构与职责

- 修改 `participant/P717/entry.py`：唯一的正式策略入口和规则推理实现。
- 创建 `participant/P717/test_rule_policy.py`：固定容量、掩码、选择、让位、角点动作和生命周期回归测试。
- 创建 `participant/P717/.gitattributes`：将 `artifacts/model_card.json` 标为 `-text`，避免 Windows 自动换行改变已登记摘要。
- 创建 `participant/P717/artifacts/model_card.json`：无学习权重的规则策略产物说明。
- 修改 `participant/P717/submission.yaml`：切换为 `rule`，登记 `model_card.json`，清空训练命令。
- 删除 `participant/P717/artifacts/policy.npz`：规则策略不再加载模板权重。
- 删除 `participant/P717/train.py`：规则提交不再携带模板 PyTorch 训练入口。
- 修改 `participant/P717/LOG.md`：记录 E002 的假设、实现、实测结果和边界。
- 修改 `participant/P717/experiments.csv`：增加 E002 的机器可读实验记录。
- 修改 `participant/P717/REPORT.md`：说明最终方法、复现命令、结果和局限。
- 修改 `participant/P717/THIRD_PARTY.md`：如实声明只使用 NumPy 和比赛接口，无外部模型或代码。
- 原样保留 `participant/P717/requirements-infer.lock` 与 `participant/P717/LICENSE`。

执行全过程位于现有功能分支 `P717-rule-baseline`，只写 `participant/P717/`。测试输出、评测输出、工作快照和新虚拟环境均放在仓库外。

### Task 1: 建立外部评测环境与不可变文件基线

**Files:**
- Verify only: `participant/P717/requirements-infer.lock`
- Verify only: `participant/P717/LICENSE`

- [ ] **Step 1: 确认分支、工作树和 Python 版本**

Run:

```powershell
git branch --show-current
git status --short
& 'C:\Users\Aurora\OneDrive\Desktop\robocup\.venv\Scripts\python.exe' --version
```

Expected: 分支为 `P717-rule-baseline`；工作树为空；Python 为 `3.12.x`。

- [ ] **Step 2: 记录两个硬性保留文件的摘要**

Run:

```powershell
Get-FileHash -Algorithm SHA256 participant/P717/requirements-infer.lock
Get-FileHash -Algorithm SHA256 participant/P717/LICENSE
```

Expected: 两个命令均输出一个 SHA-256；将输出保留在本次终端记录中，Task 7 用相同命令逐字核对。

- [ ] **Step 3: 在仓库外创建并验证正式评测环境**

Run:

```powershell
& 'C:\Users\Aurora\OneDrive\Desktop\robocup\.venv\Scripts\python.exe' -m venv 'C:\Users\Aurora\Desktop\robocup-p717-eval'
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -m pip install --no-cache-dir -r requirements-official.lock
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -m pip install --no-cache-dir hatchling editables
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -m pip install --no-cache-dir --no-build-isolation --no-deps -e .
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -m pip install --no-cache-dir pytest pytest-timeout
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' --version
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -m pip check
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -m pytest --version
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -c "from coverage_bench import get_protocol_spec; print(get_protocol_spec())"
```

Expected: Python 为 `3.12.x`，`pip check` 输出 `No broken requirements found.`，pytest 能输出版本，协议容量显示 `agent_capacity=8`、`target_capacity=8`。本机 pip HTTP 缓存曾在读取 PyPI simple-index 后持续空转，因此安装命令统一使用 `--no-cache-dir`；`hatchling` 与 `editables` 用于无隔离 editable 构建；`pytest-timeout` 负责识别仓库 pytest 配置中的 `timeout` 项。pytest 和这些开发工具只安装在仓库外环境中，不加入正式推理锁文件。

- [ ] **Step 4: 不提交环境文件**

Run:

```powershell
git status --short
```

Expected: 不出现 `.venv`、`robocup-p717-eval`、`.work`、`outputs` 或缓存文件。

### Task 2: 先写规则策略的失败测试

**Files:**
- Create: `participant/P717/test_rule_policy.py`
- Test: `participant/P717/test_rule_policy.py`

- [ ] **Step 1: 创建完整行为测试**

Create `participant/P717/test_rule_policy.py` with:

```python
import numpy as np

from entry import RuleCoveragePolicy, build_policy


CAPACITY = 8


def make_observation(*, agent_index=0, targets=None, peers=None):
    target_rows = np.zeros((CAPACITY, 3), dtype=np.float32)
    target_exists = np.zeros(CAPACITY, dtype=np.bool_)
    target_visible = np.zeros(CAPACITY, dtype=np.bool_)
    for index, vector, exists, visible in targets or []:
        target_rows[index, :2] = np.asarray(vector, dtype=np.float32)
        target_rows[index, 2] = np.float32(0.15)
        target_exists[index] = exists
        target_visible[index] = visible

    peer_rows = np.zeros((CAPACITY, 5), dtype=np.float32)
    peer_exists = np.zeros(CAPACITY, dtype=np.bool_)
    peer_visible = np.zeros(CAPACITY, dtype=np.bool_)
    for index, vector, exists, visible in peers or []:
        peer_rows[index, :2] = np.asarray(vector, dtype=np.float32)
        peer_rows[index, 4] = np.float32(0.05)
        peer_exists[index] = exists
        peer_visible[index] = visible

    return {
        "self_state": np.zeros(5, dtype=np.float32),
        "peers": peer_rows,
        "peer_exists": peer_exists,
        "peer_visible": peer_visible,
        "targets": target_rows,
        "target_exists": target_exists,
        "target_visible": target_visible,
        "agent_index": np.int64(agent_index),
        "step_index": np.int64(0),
        "time_remaining": np.float32(1.0),
    }


def assert_action(actual, expected):
    assert actual.shape == (2,)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float32))


def test_build_policy_and_lifecycle_are_stateless():
    policy = build_policy(None)
    assert isinstance(policy, RuleCoveragePolicy)
    assert policy.reset(None) is None
    assert policy.close() is None


def test_no_valid_visible_target_returns_zero_and_masks_empty_slots():
    observation = make_observation(
        targets=[
            (0, (0.1, 0.1), False, True),
            (7, (-0.1, -0.1), True, False),
        ]
    )
    assert_action(RuleCoveragePolicy().act(observation), (0.0, 0.0))


def test_nearest_valid_target_uses_corner_saturation_and_zero_component_is_positive():
    observation = make_observation(
        targets=[
            (1, (-0.4, -0.1), True, True),
            (6, (0.0, 0.2), True, True),
        ]
    )
    assert_action(RuleCoveragePolicy().act(observation), (1.0, 1.0))


def test_exact_zero_target_vector_returns_zero():
    observation = make_observation(
        targets=[(4, (0.0, 0.0), True, True)]
    )
    assert_action(RuleCoveragePolicy().act(observation), (0.0, 0.0))


def test_farther_agent_defers_first_target_and_selects_next_target():
    observation = make_observation(
        agent_index=0,
        targets=[
            (0, (0.4, 0.0), True, True),
            (5, (-0.5, 0.1), True, True),
        ],
        peers=[(1, (0.3, 0.0), True, True)],
    )
    assert_action(RuleCoveragePolicy().act(observation), (-1.0, 1.0))


def test_all_targets_deferred_returns_zero():
    observation = make_observation(
        agent_index=2,
        targets=[(3, (0.4, 0.0), True, True)],
        peers=[(1, (0.3, 0.0), True, True)],
    )
    assert_action(RuleCoveragePolicy().act(observation), (0.0, 0.0))


def test_exact_distance_tie_is_won_only_by_smaller_agent_index():
    target = [(0, (0.5, 0.0), True, True)]
    peer = [(0, (0.0, 0.0), True, True)]
    assert_action(
        RuleCoveragePolicy().act(
            make_observation(agent_index=1, targets=target, peers=peer)
        ),
        (0.0, 0.0),
    )

    higher_index_peer = [(1, (0.0, 0.0), True, True)]
    assert_action(
        RuleCoveragePolicy().act(
            make_observation(agent_index=0, targets=target, peers=higher_index_peer)
        ),
        (1.0, 1.0),
    )


def test_invisible_or_nonexistent_peer_cannot_force_deferral():
    target = [(2, (0.4, -0.2), True, True)]
    peers = [
        (1, (0.3, -0.2), False, True),
        (6, (0.3, -0.2), True, False),
    ]
    assert_action(
        RuleCoveragePolicy().act(
            make_observation(agent_index=0, targets=target, peers=peers)
        ),
        (1.0, -1.0),
    )
```

- [ ] **Step 2: 运行测试并确认它因新类尚不存在而失败**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -m pytest participant/P717/test_rule_policy.py -q -p no:cacheprovider --basetemp "$env:TEMP\robocup-p717-pytest-red"
```

Expected: collection 失败，错误包含 `cannot import name 'RuleCoveragePolicy' from 'entry'`；这证明测试确实约束新实现，而非误测旧模板。

- [ ] **Step 3: 暂存并提交红灯测试**

Run:

```powershell
git add -- participant/P717/test_rule_policy.py
git diff --cached --check
git commit -m "test(P717): 定义规则策略行为"
```

Expected: 只提交 `test_rule_policy.py`；测试仍按预期失败。

### Task 3: 实现最小无状态规则策略

**Files:**
- Modify: `participant/P717/entry.py`
- Test: `participant/P717/test_rule_policy.py`

- [ ] **Step 1: 用最小实现替换模板学习策略**

Replace `participant/P717/entry.py` with:

```python
import numpy as np


class RuleCoveragePolicy:
    """最近可见目标、按距离让位、角点饱和的无状态规则策略。"""

    def reset(self, context):
        return None

    @staticmethod
    def _corner(vector):
        if not np.any(np.abs(vector) > 0.0):
            return np.zeros(2, dtype=np.float32)
        return np.where(vector >= 0.0, 1.0, -1.0).astype(np.float32)

    def act(self, observation):
        targets = observation["targets"]
        valid_targets = observation["target_exists"] & observation["target_visible"]
        target_indices = np.flatnonzero(valid_targets)
        if target_indices.size == 0:
            return np.zeros(2, dtype=np.float32)

        target_vectors = targets[target_indices, :2].astype(np.float64)
        own_distances = np.linalg.norm(target_vectors, axis=1)
        order = np.argsort(own_distances, kind="stable")

        peers = observation["peers"]
        valid_peers = observation["peer_exists"] & observation["peer_visible"]
        peer_indices = np.flatnonzero(valid_peers)
        peer_vectors = peers[peer_indices, :2].astype(np.float64)
        own_index = int(observation["agent_index"])

        for ordered_index in order:
            target_index = int(target_indices[ordered_index])
            target_vector = targets[target_index, :2].astype(np.float64)
            own_distance = float(own_distances[ordered_index])

            beaten = False
            for peer_index, peer_vector in zip(peer_indices, peer_vectors):
                peer_distance = float(np.linalg.norm(target_vector - peer_vector))
                if peer_distance < own_distance or (
                    peer_distance == own_distance and int(peer_index) < own_index
                ):
                    beaten = True
                    break

            if not beaten:
                return self._corner(target_vector)

        return np.zeros(2, dtype=np.float32)

    def close(self):
        return None


def build_policy(context):
    return RuleCoveragePolicy()
```

- [ ] **Step 2: 运行行为测试并确认全部通过**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -m pytest participant/P717/test_rule_policy.py -q -p no:cacheprovider --basetemp "$env:TEMP\robocup-p717-pytest-green"
```

Expected: `8 passed`，且 `participant/P717/` 下没有 `__pycache__`。

- [ ] **Step 3: 检查实现没有公开案例硬编码或训练依赖**

Run:

```powershell
rg -n "225|basic-|coop-|seed|torch|policy\.npz|time_remaining|step_index" participant/P717/entry.py
```

Expected: 无匹配。策略只依据当前观测的目标、同伴、掩码和 `agent_index`。

- [ ] **Step 4: 提交策略实现**

Run:

```powershell
git add -- participant/P717/entry.py
git diff --cached --check
git commit -m "feat(P717): 实现按距离让位规则策略"
```

Expected: 该提交只包含 `entry.py`。

### Task 4: 将学习型模板清单转换为规则型提交

**Files:**
- Create: `participant/P717/.gitattributes`
- Create: `participant/P717/artifacts/model_card.json`
- Modify: `participant/P717/submission.yaml`
- Delete: `participant/P717/artifacts/policy.npz`
- Delete: `participant/P717/train.py`
- Verify only: `participant/P717/requirements-infer.lock`
- Verify only: `participant/P717/LICENSE`

- [ ] **Step 1: 创建跨平台字节稳定的说明产物**

Create `participant/P717/.gitattributes` with:

```gitattributes
artifacts/model_card.json -text
```

Create `participant/P717/artifacts/model_card.json` with the exact UTF-8 bytes represented by:

```json
{"params":0,"policy":"corner-saturate + defer-by-distance"}
```

文件末尾保留一个 LF。由于 `.gitattributes` 设置 `-text`，Git 不得在 Windows checkout 时将该 LF 转成 CRLF。

- [ ] **Step 2: 验证说明产物的固定摘要**

Run:

```powershell
git check-attr text -- participant/P717/artifacts/model_card.json
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -c "from pathlib import Path; import hashlib; p=Path('participant/P717/artifacts/model_card.json'); print(p.stat().st_size); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

Expected:

```text
participant/P717/artifacts/model_card.json: text: unset
60
11d3e61421f94b12309b1f2ece45c4816687fd1ce92f3d747449402a9b2c9c5c
```

- [ ] **Step 3: 用严格 schema 允许的字段重写清单**

Replace `participant/P717/submission.yaml` with:

```yaml
submission_schema_version: "coverage-submission/1.0"
participant_id: "P717"
protocol_version: "coverage-policy/1.0"
task_version: "coverage-task/1.0"
entrypoint: "entry:build_policy"
artifact_dir: "artifacts"
checkpoint_manifest:
  - path: "artifacts/model_card.json"
    sha256: "11d3e61421f94b12309b1f2ece45c4816687fd1ce92f3d747449402a9b2c9c5c"
    size_bytes: 60
inference_lock: "requirements-infer.lock"
method_type: "rule"
train_command: null
training_config_paths: []
report_path: "REPORT.md"
metadata:
  description: "无状态规则策略：最近可见目标、按距离让位、角点饱和；仅使用固定容量局部观测和掩码。"
```

- [ ] **Step 4: 删除不再使用的模板学习文件**

Delete exactly:

```text
participant/P717/artifacts/policy.npz
participant/P717/train.py
```

Expected: `artifacts/` 仍由已登记的 `model_card.json` 保持为真实目录；`checkpoint_manifest` 键仍存在且非空。

- [ ] **Step 5: 验证清单、文件集合和不可变文件**

Run:

```powershell
Get-ChildItem -LiteralPath participant/P717/artifacts -Force
Get-FileHash -Algorithm SHA256 participant/P717/requirements-infer.lock
Get-FileHash -Algorithm SHA256 participant/P717/LICENSE
rg -n "method_type|train_command|checkpoint_manifest|model_card|description" participant/P717/submission.yaml
rg -n "torch|torch\.load|pickle\.load|http://|https://" participant/P717 -g "*.py" -g "*.yaml"
```

Expected: `artifacts/` 只有 `model_card.json`；两个保留文件摘要与 Task 1 相同；扫描命令无匹配。

- [ ] **Step 6: 提交规则型清单迁移**

Run:

```powershell
git add -- participant/P717/.gitattributes participant/P717/artifacts/model_card.json participant/P717/submission.yaml participant/P717/artifacts/policy.npz participant/P717/train.py
git diff --cached --check
git diff --cached --name-status
git commit -m "chore(P717): 切换为规则型提交清单"
```

Expected: 新增 `.gitattributes` 与 `model_card.json`，修改 `submission.yaml`，删除 `policy.npz` 与 `train.py`；不包含其他目录。

### Task 5: 先更新不依赖成绩的提交说明

**Files:**
- Modify: `participant/P717/REPORT.md`
- Modify: `participant/P717/THIRD_PARTY.md`

- [ ] **Step 1: 将报告改为尚未报分的真实规则实现说明**

Replace `participant/P717/REPORT.md` entirely with:

````markdown
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
````

此时不要写入尚未实际运行的 `225.00` 成绩。

- [ ] **Step 2: 将第三方声明改为规则提交的真实依赖**

Replace template-learning wording in `THIRD_PARTY.md` with:

```markdown
# 第三方依赖与许可声明

本提交目录由比赛官方仓库的 `participant/_template/` 初始化，仓库采用 MIT License；保留的 `LICENSE` 为模板原文件。

P717 正式策略为本人编写的规则策略，不包含外部模型、训练权重或复制的第三方策略代码。推理仅使用比赛评测环境提供的 NumPy 与 `coverage_bench` 协议接口，依赖声明见原样保留的 `requirements-infer.lock`。
```

- [ ] **Step 3: 提交说明材料**

Run:

```powershell
git add -- participant/P717/REPORT.md participant/P717/THIRD_PARTY.md
git diff --cached --check
git commit -m "docs(P717): 说明规则策略与依赖"
```

Expected: 只提交两份说明文件，不虚构 E002 结果。

### Task 6: 在干净树上运行首次正式回归

**Files:**
- Test: `participant/P717/test_rule_policy.py`
- Validate: `participant/P717/`
- Outputs: repository-external temporary directory only

- [ ] **Step 1: 清理并证明 P717 子树没有缓存**

Run:

```powershell
Get-ChildItem -LiteralPath participant/P717 -Directory -Recurse -Force | Where-Object Name -EQ '__pycache__'
git status --short
```

Expected: 第一条无输出；工作树为空。如果发现缓存，只删除已解析且确认位于 `participant/P717/` 下的 `__pycache__`，然后再次检查。

- [ ] **Step 2: 运行完整单元测试**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -m pytest participant/P717/test_rule_policy.py -q -p no:cacheprovider --basetemp "$env:TEMP\robocup-p717-pytest-final"
```

Expected: `8 passed`。

- [ ] **Step 3: 创建唯一的仓库外输出与工作目录**

Run:

```powershell
$P717RunTag = Get-Date -Format 'yyyyMMdd-HHmmss'
$P717RunRoot = Join-Path $env:TEMP "robocup-p717-$P717RunTag"
$P717CheckOutput = Join-Path $P717RunRoot 'check'
$P717EvalOutput = Join-Path $P717RunRoot 'eval'
$env:COVERAGE_WORK_DIR = Join-Path $P717RunRoot 'work'
New-Item -ItemType Directory -Path $P717RunRoot | Out-Null
Write-Output $P717RunRoot
```

Expected: 输出一个此前不存在、位于系统临时目录的本次运行根目录。

- [ ] **Step 4: 运行提交预检**

Run:

```powershell
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' scripts/check_submission.py --submission participant/P717 --output $P717CheckOutput
if ($LASTEXITCODE -ne 0) { throw "check_submission exit=$LASTEXITCODE" }
```

Expected: 退出码 0，生成 `$P717CheckOutput\audit-report.json`。

- [ ] **Step 5: 运行公开套件**

Run:

```powershell
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' scripts/evaluate_one.py --submission participant/P717 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output $P717EvalOutput
if ($LASTEXITCODE -ne 0) { throw "evaluate_one exit=$LASTEXITCODE" }
```

Expected: 退出码 0；Windows 下 stderr 可以提示未启用阶段超时，结果 provenance 应为 `local_preview`。

- [ ] **Step 6: 用机器检查全部回归闸门**

First expose the two existing PowerShell paths without changing their values:

```powershell
$env:P717_CHECK_OUTPUT = $P717CheckOutput
$env:P717_EVAL_OUTPUT = $P717EvalOutput
```

Then run:

```powershell
@'
import csv
import json
import os
from pathlib import Path

tol = 1e-6
check_dir = Path(os.environ["P717_CHECK_OUTPUT"])
eval_dir = Path(os.environ["P717_EVAL_OUTPUT"])
audit = json.loads((check_dir / "audit-report.json").read_text(encoding="utf-8"))
result = json.loads((eval_dir / "result.json").read_text(encoding="utf-8"))
with (eval_dir / "episodes.csv").open(encoding="utf-8", newline="") as handle:
    episodes = list(csv.DictReader(handle))

assert audit["rejections"] == []
assert audit["digest"]["working_tree_dirty"] is False
assert result["status"] == "ok"
assert result["working_tree_dirty"] is False
assert result["provenance"] == "local_preview"
assert len(episodes) == 8
assert all(row["status"] == "ok" for row in episodes)
assert abs(result["performance_score"] - 225.0) <= tol
assert result["group_metrics"]["basic"]["mean_j"] >= 0.116667 - tol
assert result["group_metrics"]["cooperation"]["mean_j"] >= 0.333333 - tol
assert result["group_metrics"]["basic"]["mean_collision_rate"] == 0.0
assert result["group_metrics"]["cooperation"]["mean_collision_rate"] == 0.0
assert result["performance_score"] <= 225.0 + tol
print("all P717 public-suite gates passed")
'@ | & 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -
```

Expected: `all P717 public-suite gates passed`。任一断言失败时停止，不写成功记录，回到最早失败的单元、预检或评测阶段定位。

### Task 7: 记录 E002 的真实实验结果

**Files:**
- Modify: `participant/P717/LOG.md`
- Modify: `participant/P717/experiments.csv`
- Modify: `participant/P717/REPORT.md`

- [ ] **Step 1: 获取实际被评测的提交身份和结果字段**

Run:

```powershell
git rev-parse HEAD
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -c "import json, os, pathlib; p=pathlib.Path(os.environ['P717_EVAL_OUTPUT'])/'result.json'; r=json.loads(p.read_text(encoding='utf-8')); print(r['source_commit']); print(r['performance_score']); print(r['group_metrics']['basic']['mean_j']); print(r['group_metrics']['cooperation']['mean_j']); print(r['group_metrics']['basic']['mean_collision_rate']); print(r['group_metrics']['cooperation']['mean_collision_rate'])"
```

Expected: `source_commit` 等于当前 `HEAD`；结果依次为 `225.0`、约 `0.11666666666666667`、约 `0.3333333333333333`、`0.0`、`0.0`。

- [ ] **Step 2: 向实验 CSV 追加 E002**

使用 `apply_patch` 向 `participant/P717/experiments.csv` 追加一行。第三列 `code_commit` 直接粘贴 Step 1 已打印的 40 位 `result.json.source_commit`；其余各列依次使用以下确定值：

```text
experiment_id=E002
date=2026-09-24
method=corner_saturate_defer_by_distance
train_seed=(空)
train_steps=(空)
model_sha256=11d3e61421f94b12309b1f2ece45c4816687fd1ce92f3d747449402a9b2c9c5c
suite_id=public-suite-v1
provenance=local_preview
status=ok
performance_score=225.0
basic_mean_j=0.11666666666666667
cooperation_mean_j=0.3333333333333333
episodes_ok=8/8
notes=nearest visible target; defer to closer peer; exact-tie lower index; zero collisions
```

保存后用 `Import-Csv participant/P717/experiments.csv | Select-Object -Last 1 | Format-List` 检查字段没有因逗号错位。

- [ ] **Step 3: 在 LOG.md 记录 E002**

Append a section that states:

```markdown
## E002：角点饱和 + 按距离让位规则基线（2026-09-24）

- 目的：在只使用合法局部观测的前提下，验证最近可见目标、角点饱和和按距离让位能否达到公开套件可达上界并消除冗余碰撞。
- 方法：无训练、无模型权重；固定容量槽位先应用 `exists & visible` 掩码，目标按自身距离稳定排序；更远机器人让位，精确同距时较小编号获胜；选中目标后逐分量输出角点动作。
- 代码版本：使用 `result.json.source_commit` 的 40 位值。
- 运行环境：Windows，Python 3.12，`provenance=local_preview`；阶段超时未启用。
- 预检：退出码 0，`rejections=[]`，`digest.working_tree_dirty=false`。
- 公开测试：8/8 回合 `ok`，`performance_score=225.0`。
- 分组结果：basic `mean_j=0.11666666666666667`；cooperation `mean_j=0.3333333333333333`；两组 `mean_collision_rate=0.0`。
- 结论：在当前 4 个公开场景上达到已验证的紧上界；该数字只作为固定公开套件的回归闸门，不代表隐藏场景或组织方正式核验成绩。
```

- [ ] **Step 4: 在 REPORT.md 增加 E002 结果与局限**

使用 `apply_patch` 在 `participant/P717/REPORT.md` 的 E001 之后、复现命令之前插入以下完整章节；把“代码提交”一项改为 Step 1 已打印的真实 40 位 `source_commit`：

```markdown
## E002：角点饱和 + 按距离让位规则基线

- 日期：2026-09-24
- 代码提交：使用本次 `result.json.source_commit` 的 40 位值
- Python：3.12
- 测试套件：`public-suite-v1`
- 运行标记：`local_preview`
- 有效回合：8/8
- `performance_score`：225.0
- basic `mean_j`：0.11666666666666667
- cooperation `mean_j`：0.3333333333333333
- 两组平均碰撞率：0.0

本地公开预览证明该规则策略在当前 4 个公开场景上达到已验证的紧上界，并消除了角点饱和直冲造成的冗余碰撞。该结果不是组织方正式核验成绩。

### 局限

- 公开套件只有 4 个固定场景、每个重复 2 次，`225.00` 不能外推为隐藏场景或更大规模任务的全局上界。
- 策略在目标不可见时输出零动作；这在当前短回合公开套件上不损失得分，但更长回合或更大地图可能需要记忆与搜索。
- 精确浮点同距只用编号打破对称；编号不参与一般距离比较。
```

- [ ] **Step 5: 提交实验记录**

Run:

```powershell
git add -- participant/P717/LOG.md participant/P717/experiments.csv participant/P717/REPORT.md
git diff --cached --check
git commit -m "docs(P717): 记录225分公开回归"
```

Expected: 只提交三份真实实验材料。

### Task 8: 对最终提交树重跑全套验收并确认环境迁移条件

**Files:**
- Verify: all tracked files under `participant/P717/`
- Remove local-only: `participant/P717/**/__pycache__/` if present
- Preserve local-only: `C:\Users\Aurora\OneDrive\Desktop\robocup\.venv` until external `_monitor` probes stop referencing it

- [ ] **Step 1: 再次验证单元测试、保留文件和产物摘要**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -m pytest participant/P717/test_rule_policy.py -q -p no:cacheprovider --basetemp "$env:TEMP\robocup-p717-pytest-release"
Get-FileHash -Algorithm SHA256 participant/P717/requirements-infer.lock
Get-FileHash -Algorithm SHA256 participant/P717/LICENSE
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -c "from pathlib import Path; import hashlib; p=Path('participant/P717/artifacts/model_card.json'); assert p.stat().st_size == 60; assert hashlib.sha256(p.read_bytes()).hexdigest() == '11d3e61421f94b12309b1f2ece45c4816687fd1ce92f3d747449402a9b2c9c5c'; print('artifact digest ok')"
```

Expected: `8 passed`；两个保留文件摘要仍与 Task 1 相同；输出 `artifact digest ok`。

- [ ] **Step 2: 在新的仓库外目录重跑预检与评测**

Run:

```powershell
$env:P717_FINAL_HEAD = git rev-parse HEAD
$P717RunTag = Get-Date -Format 'yyyyMMdd-HHmmss'
$P717RunRoot = Join-Path $env:TEMP "robocup-p717-release-$P717RunTag"
$P717CheckOutput = Join-Path $P717RunRoot 'check'
$P717EvalOutput = Join-Path $P717RunRoot 'eval'
$env:COVERAGE_WORK_DIR = Join-Path $P717RunRoot 'work'
$env:P717_CHECK_OUTPUT = $P717CheckOutput
$env:P717_EVAL_OUTPUT = $P717EvalOutput
New-Item -ItemType Directory -Path $P717RunRoot | Out-Null
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' scripts/check_submission.py --submission participant/P717 --output $P717CheckOutput
if ($LASTEXITCODE -ne 0) { throw "check_submission exit=$LASTEXITCODE" }
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' scripts/evaluate_one.py --submission participant/P717 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output $P717EvalOutput
if ($LASTEXITCODE -ne 0) { throw "evaluate_one exit=$LASTEXITCODE" }
@'
import csv
import json
import os
from pathlib import Path

tol = 1e-6
check_dir = Path(os.environ["P717_CHECK_OUTPUT"])
eval_dir = Path(os.environ["P717_EVAL_OUTPUT"])
audit = json.loads((check_dir / "audit-report.json").read_text(encoding="utf-8"))
result = json.loads((eval_dir / "result.json").read_text(encoding="utf-8"))
with (eval_dir / "episodes.csv").open(encoding="utf-8", newline="") as handle:
    episodes = list(csv.DictReader(handle))

assert audit["rejections"] == []
assert audit["digest"]["working_tree_dirty"] is False
assert result["status"] == "ok"
assert result["working_tree_dirty"] is False
assert result["provenance"] == "local_preview"
assert result["source_commit"] == os.environ["P717_FINAL_HEAD"]
assert len(episodes) == 8
assert all(row["status"] == "ok" for row in episodes)
assert abs(result["performance_score"] - 225.0) <= tol
assert result["group_metrics"]["basic"]["mean_j"] >= 0.116667 - tol
assert result["group_metrics"]["cooperation"]["mean_j"] >= 0.333333 - tol
assert result["group_metrics"]["basic"]["mean_collision_rate"] == 0.0
assert result["group_metrics"]["cooperation"]["mean_collision_rate"] == 0.0
assert result["performance_score"] <= 225.0 + tol
print("final P717 public-suite gates passed")
'@ | & 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -
```

Expected: `final P717 public-suite gates passed`；此次 `result.json.source_commit` 是记录提交后的当前 HEAD，`working_tree_dirty=false`。

- [ ] **Step 3: 检查提交边界和缓存卫生**

Run:

```powershell
git status --short
git diff --name-only upstream/master...HEAD
Get-ChildItem -LiteralPath participant/P717 -Directory -Recurse -Force | Where-Object Name -EQ '__pycache__'
git ls-files participant/P717
```

Expected: 工作树为空；所有分支差异均位于 `participant/P717/`；没有 `__pycache__`；`requirements-infer.lock`、`LICENSE`、`artifacts/model_card.json` 和所有规定材料均被跟踪。

- [ ] **Step 4: 确认外部环境可完全替代仓库内旧环境**

Run:

```powershell
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' --version
& 'C:\Users\Aurora\Desktop\robocup-p717-eval\Scripts\python.exe' -m pip check
Resolve-Path -LiteralPath 'C:\Users\Aurora\OneDrive\Desktop\robocup\.venv'
Resolve-Path -LiteralPath 'C:\Users\Aurora\Desktop\robocup-p717-eval'
```

Expected: 外部环境是健康的 Python 3.12；两个路径解析为不同绝对目录。本轮保留 `C:\Users\Aurora\OneDrive\Desktop\robocup\.venv`，因为 `_monitor` 探针仍硬编码该解释器；Codex 不修改 `_monitor`。待探针维护方改用外部环境并独立验证后，再单独执行旧环境清理。

- [ ] **Step 5: 最终提交摘要检查**

Run:

```powershell
git log --oneline --decorate -10
git show --stat --oneline HEAD
git status --short
```

Expected: 最新历史包含测试、策略、规则清单、说明和真实实验记录的小提交；最终工作树为空。不要在本任务中推送、创建 PR 或向比赛仓库之外发布实现，除非用户随后明确授权。

## 自检结论

- 规格覆盖：固定容量与掩码、最近目标、距离让位、精确同距决胜、角点零分量、全零动作、`reset/act/close`、规则清单、必需产物、不可变文件、缓存卫生和全部公开回归闸门均有对应任务。
- 提交边界：所有持久修改都在 `participant/P717/`；测试和评测产物明确位于仓库外。
- 事实边界：计划只把 E001 视为现有实测；E002 必须在干净树实际得到 `225.0` 后才能写入，且始终标为 Windows `local_preview`。
- 泛化边界：策略代码不包含 `225`、公开案例名、种子或隐藏配置；225 只存在于测试验收和实验记录。
