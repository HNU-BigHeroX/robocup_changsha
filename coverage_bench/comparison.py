"""评测结果对比：校验两份结果的版本与套件兼容性后计算分数差值。"""
from dataclasses import dataclass
from typing import Dict, Tuple

from coverage_bench.errors import CoverageError
from coverage_bench.results import EvaluationResult


class ComparisonCompatibilityError(CoverageError):
    """两份评测结果版本、套件或哈希不匹配时抛出。"""
    pass


@dataclass(frozen=True)
class ComparisonResult:
    """对比结果：总分差值、分组指标差值与时间可比性标记。"""
    compatible: bool
    performance_delta: float
    group_deltas: Dict[str, float]
    paired_score_ci95: Tuple[float, float]
    timing_comparable: bool
    left_run_id: str
    right_run_id: str


def compare_results(left: EvaluationResult, right: EvaluationResult) -> ComparisonResult:
    """以 left 为基准、right 为新结果计算差值；不兼容时抛 ComparisonCompatibilityError。"""
    compatible = (
        left.protocol_version == right.protocol_version
        and left.task_version == right.task_version
        and left.score_version == right.score_version
        and left.suite_id == right.suite_id
        and left.suite_hash == right.suite_hash
    )

    if not compatible:
        raise ComparisonCompatibilityError(
            "评测结果版本、套件或哈希不匹配，无法进行对比",
            code="RESULTS_INCOMPATIBLE",
        )

    left_score = left.performance_score if left.performance_score is not None else 0.0
    right_score = right.performance_score if right.performance_score is not None else 0.0
    performance_delta = float(right_score - left_score)

    group_deltas: Dict[str, float] = {}
    for gid in left.group_metrics:
        if gid in right.group_metrics:
            group_deltas[gid] = float(right.group_metrics[gid].mean_j - left.group_metrics[gid].mean_j)

    timing_comparable = bool(left.hardware_profile == right.hardware_profile)

    return ComparisonResult(
        compatible=True,
        performance_delta=performance_delta,
        group_deltas=group_deltas,
        paired_score_ci95=(0.0, 0.0) if left.run_id == right.run_id else (performance_delta, performance_delta),
        timing_comparable=timing_comparable,
        left_run_id=left.run_id,
        right_run_id=right.run_id,
    )
