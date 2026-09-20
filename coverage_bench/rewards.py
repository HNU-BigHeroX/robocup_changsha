"""奖励计算：团队共享的覆盖收益减去碰撞惩罚。"""
from typing import Dict

from coverage_bench.envs.types import StepMetrics


def compute_reward(metrics: StepMetrics, collision_weight: float) -> Dict[str, float]:
    """由单步指标计算奖励分解（coverage / collision / team_reward）。"""
    coverage = float(metrics.coverage_rate)
    collision = float(-collision_weight * metrics.collision_rate)
    team_reward = float(coverage + collision)
    return {
        "coverage": coverage,
        "collision": collision,
        "team_reward": team_reward,
    }
