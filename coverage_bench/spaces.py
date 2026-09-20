"""动作/观测空间定义，以及把字典观测展平成固定长度向量的工具。

展平布局由 ProtocolSpec 决定（agent_capacity / target_capacity），展平版本号记录在协议中。
"""
from typing import Any

from gymnasium.spaces import Box, Dict, Discrete
import numpy as np

from coverage_bench.protocol import FloatArray, ProtocolSpec
from coverage_bench.validation import validate_observation


def make_action_space() -> Box:
    """连续 2 维动作空间，各分量取值 [-1, 1]。"""
    return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def make_observation_space(spec: ProtocolSpec) -> Dict:
    """按协议规格构造字典观测空间。"""
    return Dict({
        "self_state": Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
        "peers": Box(low=-np.inf, high=np.inf, shape=(spec.agent_capacity, 5), dtype=np.float32),
        "peer_exists": Box(low=0, high=1, shape=(spec.agent_capacity,), dtype=np.bool_),
        "peer_visible": Box(low=0, high=1, shape=(spec.agent_capacity,), dtype=np.bool_),
        "targets": Box(low=-np.inf, high=np.inf, shape=(spec.target_capacity, 3), dtype=np.float32),
        "target_exists": Box(low=0, high=1, shape=(spec.target_capacity,), dtype=np.bool_),
        "target_visible": Box(low=0, high=1, shape=(spec.target_capacity,), dtype=np.bool_),
        "agent_index": Discrete(n=spec.agent_capacity, start=0),
        "step_index": Discrete(n=spec.max_episode_steps + 1, start=0),
        "time_remaining": Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
    })


def flattened_observation_size(spec: ProtocolSpec) -> int:
    """展平观测的维度（默认规格下为 104）。"""
    return 5 + (spec.agent_capacity * 5) + spec.agent_capacity * 2 + (spec.target_capacity * 3) + spec.target_capacity * 2 + 3


def flatten_observation(observation: Any, spec: ProtocolSpec) -> FloatArray:
    """校验并把字典观测展平为固定顺序的 float32 向量。"""
    validate_observation(observation, spec)
    parts = [
        observation["self_state"].reshape(-1).astype(np.float32, copy=True),
        observation["peers"].reshape(-1).astype(np.float32, copy=True),
        observation["peer_exists"].reshape(-1).astype(np.float32, copy=True),
        observation["peer_visible"].reshape(-1).astype(np.float32, copy=True),
        observation["targets"].reshape(-1).astype(np.float32, copy=True),
        observation["target_exists"].reshape(-1).astype(np.float32, copy=True),
        observation["target_visible"].reshape(-1).astype(np.float32, copy=True),
        np.array([
            float(observation["agent_index"]),
            float(observation["step_index"]),
            float(observation["time_remaining"]),
        ], dtype=np.float32),
    ]
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
