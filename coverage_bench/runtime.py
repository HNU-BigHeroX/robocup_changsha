"""回合执行运行时：加载选手策略、逐步驱动环境并收集指标、耗时与错误分类。"""
import importlib
import time
from pathlib import Path
from typing import Callable, ContextManager

import numpy as np

from coverage_bench.deadline import (
    PHASE_ACT,
    PHASE_EPISODE,
    PHASE_LOAD,
    PhaseDeadlineExceeded,
    phase_deadline,
)
from coverage_bench.envs.factory import make_training_env
from coverage_bench.errors import CoverageError
from coverage_bench.metrics import compute_step_metrics
from coverage_bench.protocol import (
    AgentObservation,
    BuildContext,
    EpisodeContext,
    PublicTaskParams,
    ResourceLimits,
    get_protocol_spec,
)
from coverage_bench.results import EpisodeRecord, ErrorRecord
from coverage_bench.runtime_types import EpisodeRun
from coverage_bench.schedules import PolicySeedRecord
from coverage_bench.submission import ValidatedSubmission
from coverage_bench.suites import ScenarioCase
from coverage_bench.validation import validate_action

DeadlineFactory = Callable[..., ContextManager[None]]

# deadline 阶段名到结果记录 stage 字段的映射：episode 级超时沿用旧实现的 act 归属
_STAGE_BY_PHASE = {PHASE_LOAD: "initialize", PHASE_ACT: "act", PHASE_EPISODE: "act"}


def _public_task_params(case: ScenarioCase) -> PublicTaskParams:
    """把用例的 public 配置转换为协议定义的公开任务参数对象。"""
    task = case.task_config.public
    return PublicTaskParams(
        map_half_extent=task.map_half_extent,
        dt=task.dt,
        robot_radius=task.robot_radius,
        robot_mass=task.robot_mass,
        drive_force=task.drive_force,
        damping=task.damping,
        robot_max_speed=task.robot_max_speed,
        contact_force=task.contact_force,
        contact_margin=task.contact_margin,
        target_radius=task.target_radius,
        target_max_speed=task.target_max_speed,
        sense_radius=task.sense_radius,
        motion_kind=task.motion_kind,
        turn_interval_steps=tuple(task.turn_interval_steps),
        target_speed_fraction=tuple(task.target_speed_fraction),
        robot_boundary=task.robot_boundary,
        target_boundary=task.target_boundary,
    )


def load_policy(
    submission: ValidatedSubmission,
    build_seed: int,
    limits: ResourceLimits,
    deadline_factory: DeadlineFactory,
):
    """导入选手 entry 模块并在 load 阶段 deadline 内调用 build_policy 构建策略。"""
    import coverage_bench.protocol as protocol_module

    entry_dir = str(submission.root)
    import sys
    if entry_dir not in sys.path:
        sys.path.insert(0, entry_dir)
    importlib.invalidate_caches()
    try:
        entry_module = importlib.import_module("entry")
    except Exception as exc:
        raise CoverageError(f"加载 entry 模块失败: {exc}", code="LOAD_ERROR")
    if not hasattr(entry_module, "build_policy"):
        raise CoverageError("entry 模块未暴露 build_policy", code="LOAD_ERROR")
    build_ctx = BuildContext(
        spec=protocol_module.get_protocol_spec(),
        artifact_dir=submission.root / "artifacts",
        device="cpu",
        limits=limits,
        rng=np.random.default_rng(build_seed),
    )
    with deadline_factory(limits.initialization_ms, PHASE_LOAD):
        return entry_module.build_policy(build_ctx)


def run_episode(
    submission: ValidatedSubmission,
    case: ScenarioCase,
    seed_records: tuple[PolicySeedRecord, ...],
    limits: ResourceLimits,
    *,
    run_id: str,
    attempt_index: int,
    deadline_factory: DeadlineFactory = phase_deadline,
) -> EpisodeRun:
    """运行单个回合：校验种子记录、构建环境、逐 agent 加载策略并执行步循环。

    异常按阶段归类为 timeout/load_error/protocol_error/runtime_error，
    最终连同指标与耗时打包为 EpisodeRecord。
    """
    num_agents = case.task_config.num_agents
    horizon = case.task_config.horizon

    # 1. 严格校验 seed_records：数量、case_id、repeat_index 一致性与 agent_index 集合
    if len(seed_records) != num_agents:
        raise CoverageError(f"seed_records 数量不匹配: 期望 {num_agents}, 实际 {len(seed_records)}")

    first_repeat = seed_records[0].repeat_index
    agent_indices_found = set()
    seed_records_by_agent: dict[int, PolicySeedRecord] = {}
    for r in seed_records:
        if r.case_id != case.case_id:
            raise CoverageError(f"seed_record case_id 不匹配: {r.case_id} != {case.case_id}")
        if r.repeat_index != first_repeat:
            raise CoverageError(f"seed_record repeat_index 不一致: {r.repeat_index} != {first_repeat}")
        if r.agent_index in agent_indices_found:
            raise CoverageError(f"发现重复的 agent_index: {r.agent_index}")
        agent_indices_found.add(r.agent_index)
        seed_records_by_agent[r.agent_index] = r

    if agent_indices_found != set(range(num_agents)):
        raise CoverageError(f"agent_index 集合不完整: {agent_indices_found} != {set(range(num_agents))}")

    errors: list[ErrorRecord] = []
    status = "ok"
    act_timings_ms: list[float] = []
    steps_completed = 0
    total_return = 0.0
    step_coverages: list[float] = []
    step_collisions: list[float] = []
    step_full_covs: list[bool] = []
    step_collision_pairs: list[int] = []
    init_ms_sum = 0.0
    wall_start = time.perf_counter()

    env = None
    policies: list = []
    stage = "initialize"
    cur_agent_index: int | None = None
    cur_step_index: int | None = None

    try:
        # 2. 建立训练环境并按场景种子重置
        env = make_training_env(case.task_config)
        observations, _ = env.reset(seed=case.scenario_seed)
        agent_ids = list(env.agents)

        # 3. 逐个 agent 加载策略并重置，加载与 reset 耗时计入 initialization_ms_sum
        for i in range(num_agents):
            seed_rec = seed_records_by_agent[i]
            cur_agent_index = i
            cur_step_index = None
            t_load = time.perf_counter()
            policy = load_policy(submission, seed_rec.build_seed, limits, deadline_factory)
            context = EpisodeContext(
                agent_index=i,
                num_agents=num_agents,
                num_targets=case.task_config.num_targets,
                horizon=horizon,
                task=_public_task_params(case),
                policy_seed=seed_rec.policy_seed,
            )
            policy.reset(context)
            init_ms_sum += (time.perf_counter() - t_load) * 1000.0
            policies.append(policy)

        stage = "act"

        # 4. 整个步循环包在 episode 级 deadline 内，循环内每次 act 再包 act 级 deadline（嵌套结构）
        with deadline_factory(limits.episode_ms, PHASE_EPISODE):
            for step in range(horizon):
                cur_step_index = step
                actions_dict: dict[str, np.ndarray] = {}

                for i, agent_id in enumerate(agent_ids):
                    cur_agent_index = i
                    t_act = time.perf_counter()
                    with deadline_factory(limits.act_ms, PHASE_ACT, i, step):
                        raw_action = policies[i].act(observations[agent_id])
                    act_timings_ms.append((time.perf_counter() - t_act) * 1000.0)
                    # 动作校验失败按 protocol_error，error_code 取 CoverageError.code
                    validate_action(raw_action)
                    actions_dict[agent_id] = np.asarray(raw_action, dtype=np.float32)

                observations, rewards, _terminations, _truncations, _infos = env.step(actions_dict)
                total_return += float(rewards[agent_ids[0]])
                steps_completed += 1

                # 提取步级统计量
                m = compute_step_metrics(env.unwrapped._current_snapshot)
                step_coverages.append(m.coverage_rate)
                step_collisions.append(m.collision_rate)
                step_full_covs.append(bool(m.coverage_rate >= 1.0 - 1e-6))
                step_collision_pairs.append(m.collision_pairs)
    except PhaseDeadlineExceeded as exc:
        # 阶段级 deadline 超时：status=timeout，error_code 取异常自带三码
        status = "timeout"
        errors.append(ErrorRecord(
            code=exc.code,
            stage=_STAGE_BY_PHASE.get(exc.phase, "act"),
            owner="participant",
            case_id=case.case_id,
            repeat_index=first_repeat,
            agent_index=exc.agent_index,
            step_index=exc.step_index,
            message=str(exc),
        ))
    except CoverageError as exc:
        # 平台不支持阶段级 deadline 属于评测环境边界，按快速失败直接上抛，不算选手失败
        if exc.code == "DEADLINE_UNAVAILABLE":
            raise
        # 加载失败按 load_error，其余约定错误（含动作校验失败）按 protocol_error
        status = "load_error" if exc.code == "LOAD_ERROR" else "protocol_error"
        errors.append(ErrorRecord(
            code=exc.code,
            stage=stage,
            owner="participant",
            case_id=case.case_id,
            repeat_index=first_repeat,
            agent_index=cur_agent_index if stage == "act" else None,
            step_index=cur_step_index if stage == "act" else None,
            message=str(exc),
        ))
    except Exception as exc:
        # 其余策略异常按 runtime_error
        status = "runtime_error"
        errors.append(ErrorRecord(
            code="RUNTIME_ERROR",
            stage=stage,
            owner="participant",
            case_id=case.case_id,
            repeat_index=first_repeat,
            agent_index=cur_agent_index if stage == "act" else None,
            step_index=cur_step_index if stage == "act" else None,
            message=str(exc),
        ))
    finally:
        # 5. 结束阶段：逐个关闭策略并关闭环境，close 异常仅在整体仍为 ok 时提升为失败
        for i, policy in enumerate(policies):
            try:
                policy.close()
            except Exception as exc:
                if status == "ok":
                    status = "runtime_error"
                    errors.append(ErrorRecord(
                        code="RUNTIME_ERROR",
                        stage="close",
                        owner="participant",
                        case_id=case.case_id,
                        repeat_index=first_repeat,
                        agent_index=i,
                        message=f"策略关闭异常: {exc}",
                    ))
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                if status == "ok":
                    status = "runtime_error"
                    errors.append(ErrorRecord(
                        code="RUNTIME_ERROR",
                        stage="close",
                        owner="participant",
                        case_id=case.case_id,
                        repeat_index=first_repeat,
                        message=f"环境关闭异常: {exc}",
                    ))

    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    # 6. 构造 EpisodeRecord
    if status == "ok":
        return_sum_val: float | None = float(total_return)
        mean_j_val: float | None = float(total_return / horizon)
        mean_cov_val: float | None = float(np.mean(step_coverages))
        mean_col_val: float | None = float(np.mean(step_collisions))
        full_cov_frac_val: float | None = float(np.mean([1.0 if fc else 0.0 for fc in step_full_covs]))
        col_pairs_sum_val: int | None = int(sum(step_collision_pairs))
    else:
        return_sum_val = float(total_return) if steps_completed > 0 else None
        mean_j_val = float(total_return / steps_completed) if steps_completed > 0 else None
        mean_cov_val = float(np.mean(step_coverages)) if step_coverages else None
        mean_col_val = float(np.mean(step_collisions)) if step_collisions else None
        full_cov_frac_val = float(np.mean([1.0 if fc else 0.0 for fc in step_full_covs])) if step_full_covs else None
        col_pairs_sum_val = int(sum(step_collision_pairs)) if step_collision_pairs else None

    act_ms_sum = float(sum(act_timings_ms))
    act_ms_p95 = float(np.percentile(act_timings_ms, 95)) if act_timings_ms else 0.0
    act_ms_max = float(max(act_timings_ms)) if act_timings_ms else 0.0

    error_code = errors[0].code if errors else None

    record = EpisodeRecord(
        result_schema_version="coverage-result/1.0",
        run_id=run_id,
        participant_id=submission.manifest.participant_id,
        group_id=case.group_id,
        case_id=case.case_id,
        repeat_index=first_repeat,
        attempt_index=attempt_index,
        accepted=False,
        status=status,
        num_agents=num_agents,
        num_targets=case.task_config.num_targets,
        horizon=horizon,
        steps_completed=steps_completed,
        return_sum=return_sum_val,
        mean_j=mean_j_val,
        mean_coverage_rate=mean_cov_val,
        mean_collision_rate=mean_col_val,
        full_coverage_fraction=full_cov_frac_val,
        collision_pairs_sum=col_pairs_sum_val,
        act_count=len(act_timings_ms),
        act_ms_sum=act_ms_sum,
        act_ms_p95=act_ms_p95,
        act_ms_max=act_ms_max,
        initialization_ms_sum=init_ms_sum,
        wall_ms=wall_ms,
        error_code=error_code,
    )

    return EpisodeRun(
        record=record,
        errors=tuple(errors),
        worker_logs=(),
        act_timings_ms=tuple(act_timings_ms),
    )
