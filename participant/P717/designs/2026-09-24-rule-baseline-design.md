# P717 角点饱和与按距离让位规则策略设计

## 工作范围

唯一参赛工作副本为 `C:\Users\Aurora\OneDrive\Desktop\robocup`：

- `origin` 指向个人 Fork `Mortal4869/robocup_changsha`；
- 开发分支为 `P717-rule-baseline`；
- 只修改 `participant/P717/`；
- `C:\Users\Aurora\OneDrive\Desktop\deepseek robocup\robocup_changsha` 及其仓库外 `_monitor` 只作为官方源码和实验参考，不接收 P717 提交改动。

## 目标与实验证据

模板策略在本机公开预览中得到 66.67 分。仓库外单变量实验给出以下结果：

| 策略 | 本地公开预览分数 |
| --- | ---: |
| 零动作 | 83.33 |
| 最近可见目标、单位向量直冲 | 183.33 |
| 保方向方块饱和 | 208.33 |
| 角点饱和 | 221.67 |
| 角点饱和、按距离让位 | 225.00 |

可达性脚本按每步方块可达集建立机器人—目标二部图并求最大匹配，得到本次四个公开场景的乐观上界 225.00；零动作复算为 83.33，与独立零动作评测一致。P009 在四个场景的覆盖率均达到该上界，同时碰撞率为 0，因此 225.00 是当前公开套件上已经实现的紧上界。

这个数字只用于本地公开套件的回归闸门，不是隐藏场景或官方核验成绩的全局上界，也不得进入策略代码。

## 策略结构

策略采用无状态的局部规则，不使用训练模型、历史记忆或案例特征。每个机器人每步执行：

1. 使用 `target_exists & target_visible` 选出合法可见目标，忽略固定容量观测中的空槽。
2. 按自身到目标的相对距离稳定升序排列候选目标。
3. 对每个候选目标，检查全部满足 `peer_exists & peer_visible` 的可见同伴。
4. `peers[k]` 使用绝对机器人编号 `k` 索引。对目标 `j`，同伴到目标的相对向量为 `peers[k, :2] - targets[j, :2]`。
5. 若同伴到目标的距离严格小于自身距离，则自身让出该目标。
6. 若距离浮点值完全相等，仅用较小 `agent_index` 决胜，防止双方同时抢占或同时让出；编号不参与正常距离优先级。
7. 选择第一个没有被更近同伴占优的目标。
8. 没有可见目标，或所有可见目标均被更近同伴占优时，输出零动作。
9. 选中目标后执行角点饱和：若目标二维向量整体为零则输出零动作；否则每个非负分量输出 `+1`，负分量输出 `-1`。

角点饱和故意利用动作空间是逐分量 `[-1, 1]` 方块，而非单位圆盘。二维非轴向动作范数可达到 `sqrt(2)`。该规则保留 P009 已验证的零分量边界：当目标向量不是整体零向量时，单个恰为零的分量取 `+1`。

## 明确排除的机制

公开套件单变量实验表明，下列机制没有收益或产生净负贡献，因此不进入本基线：

- 固定 `agent_index % num_targets` 目标绑定；
- 粘性认领、TTL、目标丢失记忆和目标速度外推；
- 盲步朝地图中心或历史目标移动；
- 速度前馈、到达减速、逆动力学控制；
- 人为设置 `v_cap = drive_force / robot_mass * dt / damping`；
- 按机器人编号而不是距离决定正常让位顺序。

其中 `agent_index` 只保留为“距离完全相等”的对称性决胜键。

## 物理与指标边界

- 动作逐分量限制在 `[-1, 1]`，动作集合是方块，二维范数上限为 `sqrt(2)`。
- MPE2 先用旧速度更新位置，再更新速度，因此动作对位置存在一拍延迟。
- 公开参数下第 10 步单轴最大位移约为 0.249，对角最大位移约为 `0.249 * sqrt(2) = 0.352`。
- 覆盖判定只检查机器人中心与目标中心的距离是否不超过 `target_radius`；`robot_radius` 不参与覆盖半径。
- `robot_radius` 参与机器人碰撞判定。
- 公开套件正式计分为 3v3，但实现必须遵守容量为 8 的固定形状和掩码，不把机器人或目标数量写死为 3。

## 接口与健壮性

`entry.py` 导出 `build_policy(context)`，返回对象实现：

- `reset(context)`：策略无跨步状态，但方法必须存在；
- `act(observation)`：始终返回形状 `(2,)`、类型 `np.float32`、逐分量位于 `[-1, 1]` 的有限动作；
- `close()`：必须存在并可安全调用，否则可能导致顶层评测状态失败。

异常观测处理：

- 只处理同时满足 `exists` 和 `visible` 的槽位；
- 自身索引槽不得作为同伴；
- 非有限或形状错误的关键数组安全回退为零动作；
- 目标相对向量整体为零时返回零动作；
- 不读取案例编号、场景种子、隐藏配置、官方评测器内部状态或文件系统中的评测结果。

## 提交文件设计

全部变更限制在 `participant/P717/`：

- 用最小规则实现替换 `entry.py`；
- 新增 `test_rule_policy.py`；
- 删除模板 `train.py`；
- 删除模板 `artifacts/policy.npz`；
- 新增 `artifacts/model_card.json`，内容声明规则名称和参数量为 0；
- 更新 `submission.yaml`、`LOG.md`、`experiments.csv`、`REPORT.md` 和 `THIRD_PARTY.md`；
- `requirements-infer.lock` 与 `LICENSE` 必须保持原样。

`artifacts/` 是必需目录，但 Git 无法跟踪空目录，因此不得使用“空目录加空清单”的最终提交方式。也不得使用 `.gitkeep`：未登记会触发“产物未登记”，登记后又会因无允许后缀而触发 `ARTIFACT_FORMAT_NOT_ALLOWED`。

使用白名单格式 `artifacts/model_card.json`，示例语义为：

```json
{"policy":"corner-saturate + defer-by-distance","params":0}
```

该文件不参与策略运行，但必须登记在 `checkpoint_manifest`。哈希和字节数只能在文件内容最终确定后计算。

`submission.yaml` 采用：

- `method_type: rule`；
- `train_command: null`；
- `training_config_paths: []`；
- `checkpoint_manifest` 必须保留且包含 `model_card.json` 条目，不能省略；
- `metadata.description` 写入规则策略说明；
- 不新增 schema 未定义的顶层字段，因为模型使用 `extra="forbid"`。

删除 `train.py` 的原因是提交语义一致性和人工审核卫生。`torch` 命中属于 `scan_hits`，本身不是自动硬拒绝；不得为了消除命中而使用混淆或动态导入，否则可能触发真正的硬拒绝。

## 测试设计

单元测试至少覆盖：

- 无可见目标返回零动作；
- `exists=false` 或 `visible=false` 的空槽不参与选择；
- 从多个目标中选择最近目标；
- 可见同伴离目标更近时，自身让出该目标并尝试下一个目标；
- 所有目标均被更近同伴占优时返回零动作；
- 自身更近时不因同伴编号更低而让位；
- 距离完全相等时较小 `agent_index` 获得目标；
- `peers[k]` 按绝对编号解释，自身槽不会参与；
- 角点饱和输出逐分量为 `+1/-1`；
- 非整体零向量中的零分量取 `+1`；
- 整体零向量返回零动作；
- 输出始终为有限的 `(2,) float32`；
- `reset()` 和 `close()` 可重复安全调用；
- 使用不同的存在掩码测试 3v3 之外的容量布局，不把有效数量写死。

所有 Python 测试都设置 `PYTHONDONTWRITEBYTECODE=1`，避免在提交子树产生 `__pycache__`。

## 实验与验收

零动作、确定性随机动作和其他对照策略只放在仓库外临时目录。所有评测使用相同的 `configs/public-suite-v1.yaml` 和 `configs/public-seeds-v1.json`，每次使用全新的 `--output` 目录。

P717 最终公开预览采用以下回归闸门，比较浮点值时使用 `1e-6` 容差：

- 预检退出码为 0，`rejections` 为空；
- 顶层 `result.json.status == "ok"`；
- 8/8 回合状态为 `ok`；
- `abs(performance_score - 225.00) <= 1e-6`；
- `basic.mean_j >= 0.116667 - 1e-6`；
- `cooperation.mean_j >= 0.333333 - 1e-6`；
- 基础组与协作组 `mean_collision_rate == 0.0`；
- `audit-report.json.digest.working_tree_dirty == false`；
- `result.json.working_tree_dirty == false`；
- Windows 结果明确标记为 `provenance=local_preview`，不冒充官方核验结果。

`performance_score > 225.00 + 1e-6` 在当前公开配置下视为异常信号，应检查套件、种子、代码树和是否读取了禁止状态，而不是当作更好成绩。`performance_score_ci95` 不作为本地回归判断依据。

内部提交卫生目标为删除模板产生的 `torch` 扫描命中；但 `scan_hits` 不等同于硬拒绝。如仍有任何命中，必须逐条人工检查并在报告中解释，不能只看 `rejections`。

## 环境与提交卫生

- 开发和评测虚拟环境建在仓库外；现有仓库根目录 `.venv` 先保留，待仓库外替代环境验证通过并获得用户确认后再处理。
- 评测器工作目录和输出目录放在仓库外临时目录。
- 提交前删除 `participant/P717/` 下所有 `__pycache__`、`*.pyc` 和 `*.pyo`。
- 最终检查 `git status --short --ignored -- participant/P717`，并以 audit/report 中的 `working_tree_dirty=false` 为准。
- `requirements-infer.lock` 和 `LICENSE` 与模板做 SHA-256 对比，确认字节级不变。
- 只暂存 `participant/P717/` 中计划内文件。
- 功能分支验证通过后再推送；最终 PR 只在组织方统一提交窗口开放后创建。

## 泛化实验边界

先实现并锁定 225.00 的公开基线。更大规模或隐藏分布的泛化研究作为后续独立实验，例如 4v5、5v7、较长时域下的盲区搜索或短期记忆；这些实验必须单变量比较，不能破坏公开基线，也不能把公开案例坐标、编号、种子或 225 分门槛写入策略逻辑。
