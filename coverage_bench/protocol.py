"""选手策略与环境之间的协议契约：协议规格、公共任务参数、观测/动作批量类型与资源限制。

所有 dataclass 均冻结，字段即评测系统与选手代码之间的稳定接口。
"""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
AgentId = str


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    """协议级常量：容量、动作维度、缩放尺度、回合上限与展平版本。"""

    protocol_version: str
    task_version: str
    agent_capacity: int
    target_capacity: int
    action_dim: int
    position_scale: float
    velocity_scale: float
    max_episode_steps: int
    flatten_version: str


@dataclass(frozen=True, slots=True)
class PublicTaskParams:
    """对选手公开的任务物理与目标运动参数（不允许隐藏信息）。"""

    map_half_extent: float
    dt: float
    robot_radius: float
    robot_mass: float
    drive_force: float
    damping: float
    robot_max_speed: float
    contact_force: float
    contact_margin: float
    target_radius: float
    target_max_speed: float
    sense_radius: float
    motion_kind: str
    turn_interval_steps: tuple[int, int]
    target_speed_fraction: tuple[float, float]
    robot_boundary: str
    target_boundary: str


@dataclass(frozen=True, slots=True)
class EpisodeContext:
    """reset 时下发给单个 agent 的回合上下文。"""

    agent_index: int
    num_agents: int
    num_targets: int
    horizon: int
    task: PublicTaskParams
    policy_seed: int


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """评测运行时的资源限制（时间、内存、进程、日志等）。"""

    initialization_ms: int = 30000
    act_ms: int = 1000
    episode_ms: int = 120000
    evaluation_ms: int = 600000
    close_ms: int = 5000
    memory_mib_per_worker: int = 512
    artifact_bytes_total: int = 104857600
    submission_bytes_total: int = 268435456
    scratch_bytes_per_worker: int = 67108864
    log_bytes_per_worker: int = 1048576
    max_processes_per_worker: int = 32
    cpu_threads_per_worker: int = 1
    ipc_message_bytes: int = 65536


@dataclass(frozen=True, slots=True)
class BuildContext:
    """构建策略实例时提供的构建上下文。"""

    spec: ProtocolSpec
    artifact_dir: Path
    device: str
    limits: ResourceLimits
    rng: np.random.Generator


class AgentObservation(TypedDict):
    """单个 agent 的局部观测：自身状态、同伴/目标的相对量与可见性标记。"""

    self_state: FloatArray
    peers: FloatArray
    peer_exists: BoolArray
    peer_visible: BoolArray
    targets: FloatArray
    target_exists: BoolArray
    target_visible: BoolArray
    agent_index: np.int64
    step_index: np.int64
    time_remaining: np.float32


from typing import Any, Protocol


class Policy(Protocol):
    """选手策略须实现的协议接口：reset / act / close。"""

    def reset(self, context: EpisodeContext) -> None:
        ...

    def act(self, observation: AgentObservation) -> FloatArray:
        ...

    def close(self) -> None:
        ...


class RewardBreakdown(TypedDict):
    """奖励分解：覆盖项、碰撞项与团队总奖励。"""

    coverage: float
    collision: float
    team_reward: float


class TrainingInfo(TypedDict):
    """step 后下发给训练侧的信息。"""

    step_index: int
    metrics: Any
    reward_terms: RewardBreakdown | None


ObservationBatch = dict[AgentId, AgentObservation]
InfoBatch = dict[AgentId, TrainingInfo]
ActionBatch = dict[AgentId, FloatArray]


def get_protocol_spec() -> ProtocolSpec:
    """从打包的 development-protocol.json 读取协议规格。"""
    spec_path = Path(__file__).resolve().parent / "data" / "development-protocol.json"
    content = json.loads(spec_path.read_text(encoding="utf-8"))
    return ProtocolSpec(**content)
