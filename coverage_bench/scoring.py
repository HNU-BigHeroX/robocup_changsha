"""评分配置解析与成绩聚合：组指标、bootstrap 置信区间与 performance_score 计算。"""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np
import yaml

from coverage_bench.errors import CoverageError
from coverage_bench.results import EpisodeRecord, GroupMetrics
from coverage_bench.suites import EvaluationSuite


@dataclass(frozen=True, slots=True)
class ScoreGroup:
    """单个评分组的权重与随机/学习锚点（用于归一化到 0-1000 分）。"""
    group_id: str
    weight: float
    random_anchor: float
    learning_anchor: float
    min_anchor_gap: float


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """评分配置：分组锚点、bootstrap 参数与近平局判定阈值。"""
    score_version: str
    groups: tuple[ScoreGroup, ...]
    bootstrap_resamples: int
    bootstrap_seed: int
    confidence_level: float
    near_tie_threshold: float
    supplementary_suite_id: str


def _config_int(value: Any, field: str, origin: str) -> int:
    if type(value) is not int:
        raise CoverageError(f"评分配置字段必须为整数 {field}: {origin}", code="SCORING_CONFIG_INVALID")
    return value


def _config_float(value: Any, field: str, origin: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CoverageError(f"评分配置字段必须为数值 {field}: {origin}", code="SCORING_CONFIG_INVALID")
    if not math.isfinite(value):
        raise CoverageError(f"评分配置字段必须为有限数值 {field}: {origin}", code="SCORING_CONFIG_INVALID")
    return float(value)


def _config_str(value: Any, field: str, origin: str) -> str:
    if not isinstance(value, str) or not value:
        raise CoverageError(f"评分配置字段必须为非空字符串 {field}: {origin}", code="SCORING_CONFIG_INVALID")
    return value


def scoring_config_from_data(data: Any, origin: str) -> ScoringConfig:
    """从映射数据解析评分配置，字段集合与类型不符即抛 CoverageError。"""
    if not isinstance(data, dict):
        raise CoverageError(f"评分配置根结构必须为映射: {origin}", code="SCORING_CONFIG_INVALID")
    required = {
        "score_version", "groups", "bootstrap_resamples", "bootstrap_seed",
        "confidence_level", "near_tie_threshold", "supplementary_suite_id",
    }
    if set(data.keys()) != required:
        raise CoverageError(
            f"评分配置字段集合不符: {origin}，多余 {set(data.keys()) - required}，缺失 {required - set(data.keys())}",
            code="SCORING_CONFIG_INVALID",
        )
    groups_raw = data["groups"]
    if not isinstance(groups_raw, list) or not groups_raw:
        raise CoverageError(f"评分配置 groups 必须为非空列表: {origin}", code="SCORING_CONFIG_INVALID")
    groups = []
    for entry in groups_raw:
        if not isinstance(entry, dict) or set(entry.keys()) != {
            "group_id", "weight", "random_anchor", "learning_anchor", "min_anchor_gap",
        }:
            raise CoverageError(f"评分分组字段集合不符: {origin}, {entry}", code="SCORING_CONFIG_INVALID")
        groups.append(ScoreGroup(
            group_id=_config_str(entry["group_id"], "groups.group_id", origin),
            weight=_config_float(entry["weight"], "groups.weight", origin),
            random_anchor=_config_float(entry["random_anchor"], "groups.random_anchor", origin),
            learning_anchor=_config_float(entry["learning_anchor"], "groups.learning_anchor", origin),
            min_anchor_gap=_config_float(entry["min_anchor_gap"], "groups.min_anchor_gap", origin),
        ))
    return ScoringConfig(
        score_version=_config_str(data["score_version"], "score_version", origin),
        groups=tuple(groups),
        bootstrap_resamples=_config_int(data["bootstrap_resamples"], "bootstrap_resamples", origin),
        bootstrap_seed=_config_int(data["bootstrap_seed"], "bootstrap_seed", origin),
        confidence_level=_config_float(data["confidence_level"], "confidence_level", origin),
        near_tie_threshold=_config_float(data["near_tie_threshold"], "near_tie_threshold", origin),
        supplementary_suite_id=_config_str(
            data["supplementary_suite_id"], "supplementary_suite_id", origin,
        ),
    )


def load_scoring_config(path: Path) -> ScoringConfig:
    """从 YAML 文件加载评分配置。"""
    target = Path(path)
    if not target.is_file():
        raise CoverageError(f"评分配置文件不存在: {path}")
    return scoring_config_from_data(yaml.safe_load(target.read_text(encoding="utf-8")), str(target))


def default_scoring_config(suite: EvaluationSuite) -> ScoringConfig:
    # 按套件推导缺省评分配置：组等权、锚点 [0, 1]、固定 bootstrap 参数。
    # 容器内评测入口与宿主侧复算必须使用同一实现，保证评分口径一致
    num_groups = len(suite.groups)
    weight = 1.0 / max(1, num_groups)
    groups = tuple(
        ScoreGroup(
            group_id=g.group_id,
            weight=weight,
            random_anchor=0.0,
            learning_anchor=1.0,
            min_anchor_gap=0.01,
        )
        for g in suite.groups
    )
    return ScoringConfig(
        score_version=suite.score_version or "coverage-score/1.0",
        groups=groups,
        bootstrap_resamples=1000,
        bootstrap_seed=81273,
        confidence_level=0.95,
        near_tie_threshold=1.0,
        supplementary_suite_id="acceptance-supplementary",
    )


def _validate_scoring_config(groups: dict[str, GroupMetrics], scoring: ScoringConfig) -> None:
    """校验评分配置与待评分组一致（组集合、权重和为 1、锚点间距）。"""
    config_group_ids = [g.group_id for g in scoring.groups]
    if len(config_group_ids) != len(set(config_group_ids)):
        raise CoverageError("评分配置中包含重复的 group_id")
    
    if set(config_group_ids) != set(groups.keys()):
        raise CoverageError(f"待评分组与配置组不匹配: {set(groups.keys())} != {set(config_group_ids)}")
        
    weights_sum = 0.0
    for g in scoring.groups:
        if not math.isfinite(g.weight) or g.weight < 0.0:
            raise CoverageError(f"组权重必须为非负有限数值: {g.weight}")
        weights_sum += g.weight
        
        if not math.isfinite(g.random_anchor):
            raise CoverageError(f"random_anchor 必须为有限数值: {g.random_anchor}")
        if not math.isfinite(g.learning_anchor):
            raise CoverageError(f"learning_anchor 必须为有限数值: {g.learning_anchor}")
        if not math.isfinite(g.min_anchor_gap) or g.min_anchor_gap <= 0.0:
            raise CoverageError(f"min_anchor_gap 必须为正数: {g.min_anchor_gap}")
            
        if g.learning_anchor - g.random_anchor < g.min_anchor_gap:
            raise CoverageError(
                f"锚点间距过小: {g.learning_anchor} - {g.random_anchor} < {g.min_anchor_gap}"
            )
            
    if abs(weights_sum - 1.0) > 1e-12:
        raise CoverageError(f"组权重之和与 1.0 的偏差超过允许容差 1e-12: 偏差为 {abs(weights_sum - 1.0)}")


def calculate_performance_score(groups: dict[str, GroupMetrics], scoring: ScoringConfig) -> float:
    """按组权重将各组 mean_j 锚点归一化后加权求和得总分。"""
    _validate_scoring_config(groups, scoring)
    
    total_score = 0.0
    for g in scoring.groups:
        metric = groups[g.group_id]
        normalized = (metric.mean_j - g.random_anchor) / (g.learning_anchor - g.random_anchor)
        total_score += 1000.0 * g.weight * normalized
        
    return float(total_score)


def aggregate_episodes(
    records: tuple[EpisodeRecord, ...],
    suite: EvaluationSuite,
    scoring: ScoringConfig,
    timings: Optional[np.ndarray | Sequence[float] | dict[str, Sequence[float]]] = None,
) -> dict[str, GroupMetrics]:
    """校验回合记录完整后聚合各组指标（场景均值、动作耗时分位数、bootstrap CI）。"""
    if not records:
        raise CoverageError("回合记录列表为空")
        
    first_record = records[0]
    expected_run_id = first_record.run_id
    expected_participant = first_record.participant_id
    
    records_map: dict[tuple[str, str, int], EpisodeRecord] = {}
    for r in records:
        if r.run_id != expected_run_id:
            raise CoverageError(f"回合记录包含不一致的 run_id: {r.run_id} != {expected_run_id}")
        if r.participant_id != expected_participant:
            raise CoverageError(f"回合记录包含不一致的 participant_id: {r.participant_id} != {expected_participant}")
        if not r.accepted:
            raise CoverageError(f"未被接纳的回合记录不能参与评分: case={r.case_id}, repeat={r.repeat_index}")
        if r.status != "ok":
            raise CoverageError(f"非 ok 状态的回合记录不能参与评分: status={r.status}")
            
        key = (r.group_id, r.case_id, r.repeat_index)
        if key in records_map:
            raise CoverageError(f"发现重复的回合记录: {key}")
        records_map[key] = r

    # 验证套件所有预期回合
    expected_keys: set[tuple[str, str, int]] = set()
    suite_groups_map = {g.group_id: g for g in suite.groups}
    for group in suite.groups:
        for case in group.cases:
            for rep in range(group.policy_repeats):
                k = (group.group_id, case.case_id, rep)
                expected_keys.add(k)
                if k not in records_map:
                    raise CoverageError(f"套件预期回合记录缺失: {k}")
                rec = records_map[k]
                if rec.horizon != case.task_config.horizon:
                    raise CoverageError(f"回合步数与套件配置不符: {rec.horizon} != {case.task_config.horizon}")
                if rec.num_agents != case.task_config.num_agents:
                    raise CoverageError(f"智能体数量与套件配置不符: {rec.num_agents} != {case.task_config.num_agents}")

    if set(records_map.keys()) != expected_keys:
        extra_keys = set(records_map.keys()) - expected_keys
        raise CoverageError(f"存在多余的回合记录: {extra_keys}")

    # 拆分全量动作耗时数组
    group_timings_map: dict[str, list[float]] = {}
    if timings is not None:
        if isinstance(timings, dict):
            for gid, arr in timings.items():
                group_timings_map[gid] = [float(x) for x in arr]
        else:
            flat_arr = np.asarray(timings, dtype=np.float64).ravel()
            offset = 0
            for g in suite.groups:
                g_timings: list[float] = []
                for c in g.cases:
                    for rep in range(g.policy_repeats):
                        rec = records_map.get((g.group_id, c.case_id, rep))
                        c_cnt = rec.act_count if rec is not None else 0
                        if offset + c_cnt <= len(flat_arr):
                            g_timings.extend(flat_arr[offset : offset + c_cnt].tolist())
                            offset += c_cnt
                group_timings_map[g.group_id] = g_timings

    # 计算各组指标
    result_groups: dict[str, GroupMetrics] = {}
    
    # 为保证可复现的 bootstrap，按组名确定顺序
    for group_index, group in enumerate(suite.groups):
        num_scenarios = len(group.cases)
        num_repeats = group.policy_repeats
        num_episodes = num_scenarios * num_repeats
        
        # 提取各场景均值
        scenario_mean_j_list: list[float] = []
        scenario_mean_return_list: list[float] = []
        scenario_mean_cov_list: list[float] = []
        scenario_mean_col_list: list[float] = []
        scenario_full_cov_list: list[float] = []
        
        group_act_ms_sum = 0.0
        group_act_count = 0
        group_act_p95_list: list[float] = []
        group_act_max_list: list[float] = []

        for case in group.cases:
            case_records = [records_map[(group.group_id, case.case_id, rep)] for rep in range(num_repeats)]
            scenario_mean_j_list.append(float(np.mean([r.mean_j for r in case_records])))
            scenario_mean_return_list.append(float(np.mean([r.return_sum for r in case_records])))
            scenario_mean_cov_list.append(float(np.mean([r.mean_coverage_rate for r in case_records])))
            scenario_mean_col_list.append(float(np.mean([r.mean_collision_rate for r in case_records])))
            scenario_full_cov_list.append(float(np.mean([r.full_coverage_fraction for r in case_records])))
            
            for r in case_records:
                group_act_ms_sum += r.act_ms_sum
                group_act_count += r.act_count
                group_act_p95_list.append(r.act_ms_p95)
                group_act_max_list.append(r.act_ms_max)

        mean_j = float(np.mean(scenario_mean_j_list))
        mean_return = float(np.mean(scenario_mean_return_list))
        mean_cov = float(np.mean(scenario_mean_cov_list))
        mean_col = float(np.mean(scenario_mean_col_list))
        full_cov = float(np.mean(scenario_full_cov_list))
        
        # 精确动作分位数计算
        if group.group_id in group_timings_map and len(group_timings_map[group.group_id]) > 0:
            arr_t = np.array(group_timings_map[group.group_id], dtype=np.float64)
            act_ms_mean = float(np.mean(arr_t))
            act_ms_p95 = float(np.percentile(arr_t, 95))
            act_ms_max = float(np.max(arr_t))
        else:
            act_ms_mean = float(group_act_ms_sum / max(1, group_act_count))
            act_ms_p95 = float(np.percentile(group_act_p95_list, 95)) if group_act_p95_list else 0.0
            act_ms_max = float(max(group_act_max_list)) if group_act_max_list else 0.0

        # Bootstrap 置信区间
        ci95_j: tuple[float, float] | None = None
        if num_scenarios >= 2:
            seed_seq = np.random.SeedSequence(scoring.bootstrap_seed, spawn_key=(group_index,))
            rng = np.random.default_rng(seed_seq)
            scenario_array = np.array(scenario_mean_j_list, dtype=np.float64)
            resampled_indices = rng.integers(0, num_scenarios, size=(scoring.bootstrap_resamples, num_scenarios))
            resampled_means = np.mean(scenario_array[resampled_indices], axis=1)
            
            alpha = (1.0 - scoring.confidence_level) / 2.0
            low = float(np.percentile(resampled_means, 100.0 * alpha))
            high = float(np.percentile(resampled_means, 100.0 * (1.0 - alpha)))
            ci95_j = (low, high)

        result_groups[group.group_id] = GroupMetrics(
            num_scenarios=num_scenarios,
            num_repeats=num_repeats,
            num_episodes=num_episodes,
            mean_return=mean_return,
            mean_j=mean_j,
            mean_coverage_rate=mean_cov,
            mean_collision_rate=mean_col,
            full_coverage_fraction=full_cov,
            ci95_j=ci95_j,
            act_ms_mean=act_ms_mean,
            act_ms_p95=act_ms_p95,
            act_ms_max=act_ms_max,
        )

    return result_groups


def select_supplementary_groups(scores: dict[str, float], threshold: float) -> list[tuple[str, ...]]:
    """按分数排序找出差距不超过 threshold 的近平局组（每组至少 2 名选手）。"""
    if not scores:
        return []
        
    sorted_items = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    connected_groups: list[list[str]] = []
    current_group: list[str] = [sorted_items[0][0]]
    
    for i in range(1, len(sorted_items)):
        prev_name, prev_score = sorted_items[i - 1]
        curr_name, curr_score = sorted_items[i]
        if (prev_score - curr_score) <= threshold:
            current_group.append(curr_name)
        else:
            if len(current_group) >= 2:
                connected_groups.append(current_group)
            current_group = [curr_name]
            
    if len(current_group) >= 2:
        connected_groups.append(current_group)
        
    return [tuple(g) for g in connected_groups]
