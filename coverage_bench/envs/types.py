"""环境内部数据类型：世界快照、单步指标与场景状态。"""
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from coverage_bench.config import TaskConfig


@dataclass
class WorldSnapshot:
    """某一仿真时刻的全局世界状态（观测与指标计算只读该快照）。"""

    robot_positions: NDArray[np.float64]
    robot_velocities: NDArray[np.float64]
    robot_radii: NDArray[np.float64]
    target_positions: NDArray[np.float64]
    target_velocities: NDArray[np.float64]
    target_radii: NDArray[np.float64]
    step_index: int


@dataclass
class StepMetrics:
    """单步评测指标：匹配目标数、碰撞统计、覆盖率与碰撞率等。"""

    matched_targets: int
    collision_agents: int
    collision_pairs: int
    coverage_rate: float
    collision_rate: float
    full_coverage: bool


from coverage_bench.protocol import RewardBreakdown


@dataclass
class ScenarioState:
    """环境的可变场景状态：机器人/目标实体数据、目标转向调度与随机源。

    mpe_world 为锁定的 MPE2 仿真世界；为 None 时 physics 走内置简化积分。
    """

    config: TaskConfig
    robot_positions: NDArray[np.float64]
    robot_velocities: NDArray[np.float64]
    robot_radii: NDArray[np.float64]
    target_positions: NDArray[np.float64]
    target_velocities: NDArray[np.float64]
    target_radii: NDArray[np.float64]
    step_index: int
    target_next_turn: NDArray[np.int64]
    target_rng: np.random.Generator
    mpe_world: Any | None = None
