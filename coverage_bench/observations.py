"""逐 agent 的局部观测编码，以及训练/评测用的全局状态编码。

观测中的位置、速度、半径均按协议的 position_scale / velocity_scale 归一化，远处实体只给存在/可见标记。
"""
from math import dist
from typing import Any

import numpy as np

from coverage_bench.config import TaskConfig
from coverage_bench.envs.types import WorldSnapshot
from coverage_bench.protocol import AgentObservation, FloatArray, ProtocolSpec


class _ObservationDict(dict):
    """普通 dict 观测的子类，支持 np.asarray() 直接展平（便于策略侧与向量化环境使用）。"""

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        parts = [
            np.asarray(self["self_state"], dtype=np.float32).reshape(-1),
            np.asarray(self["peers"], dtype=np.float32).reshape(-1),
            np.asarray(self["peer_exists"], dtype=np.float32).reshape(-1),
            np.asarray(self["peer_visible"], dtype=np.float32).reshape(-1),
            np.asarray(self["targets"], dtype=np.float32).reshape(-1),
            np.asarray(self["target_exists"], dtype=np.float32).reshape(-1),
            np.asarray(self["target_visible"], dtype=np.float32).reshape(-1),
            np.array([
                float(self["agent_index"]),
                float(self["step_index"]),
                float(self["time_remaining"]),
            ], dtype=np.float32),
        ]
        arr = np.concatenate(parts)
        if dtype is not None:
            arr = arr.astype(dtype, copy=copy or False)
        return arr


def observe_agent(
    world: WorldSnapshot,
    agent_index: int,
    config: TaskConfig,
    spec: ProtocolSpec,
) -> AgentObservation:
    """编码 agent_index 号机器人的局部观测（感知半径外的实体字段保持零）。"""
    L = float(spec.position_scale)
    V = float(spec.velocity_scale)
    A = spec.agent_capacity
    B = spec.target_capacity
    N = len(world.robot_positions)
    M = len(world.target_positions)
    sense_r = float(config.public.sense_radius)

    self_pos = world.robot_positions[agent_index]
    self_vel = world.robot_velocities[agent_index]
    self_radius = float(world.robot_radii[agent_index])

    self_state = np.array([
        self_pos[0] / L,
        self_pos[1] / L,
        self_vel[0] / V,
        self_vel[1] / V,
        self_radius / L,
    ], dtype=np.float32)

    peers = np.zeros((A, 5), dtype=np.float32)
    peer_exists = np.zeros(A, dtype=np.bool_)
    peer_visible = np.zeros(A, dtype=np.bool_)

    for i in range(min(N, A)):
        if i != agent_index:
            peer_exists[i] = True
            dx = float(world.robot_positions[i, 0] - self_pos[0])
            dy = float(world.robot_positions[i, 1] - self_pos[1])
            d = dist((float(self_pos[0]), float(self_pos[1])), (float(world.robot_positions[i, 0]), float(world.robot_positions[i, 1])))
            if d <= sense_r:
                peer_visible[i] = True
                dvx = float(world.robot_velocities[i, 0] - self_vel[0])
                dvy = float(world.robot_velocities[i, 1] - self_vel[1])
                pr = float(world.robot_radii[i])
                peers[i] = [dx / L, dy / L, dvx / V, dvy / V, pr / L]

    targets = np.zeros((B, 3), dtype=np.float32)
    target_exists = np.zeros(B, dtype=np.bool_)
    target_visible = np.zeros(B, dtype=np.bool_)

    for j in range(min(M, B)):
        target_exists[j] = True
        dx = float(world.target_positions[j, 0] - self_pos[0])
        dy = float(world.target_positions[j, 1] - self_pos[1])
        d = dist((float(self_pos[0]), float(self_pos[1])), (float(world.target_positions[j, 0]), float(world.target_positions[j, 1])))
        if d <= sense_r:
            target_visible[j] = True
            tr = float(world.target_radii[j])
            targets[j] = [dx / L, dy / L, tr / L]

    horizon = config.horizon
    step = world.step_index
    time_rem = np.float32((horizon - step) / horizon)

    return _ObservationDict(
        self_state=self_state,
        peers=peers,
        peer_exists=peer_exists,
        peer_visible=peer_visible,
        targets=targets,
        target_exists=target_exists,
        target_visible=target_visible,
        agent_index=np.int64(agent_index),
        step_index=np.int64(step),
        time_remaining=time_rem,
    )


def encode_global_state(
    world: WorldSnapshot,
    config: TaskConfig,
    spec: ProtocolSpec,
) -> FloatArray:
    """编码全局状态向量：机器人表 + 存在标记 + 目标表 + 存在标记 + 剩余时间。"""
    L = float(spec.position_scale)
    V = float(spec.velocity_scale)
    A = spec.agent_capacity
    B = spec.target_capacity
    N = len(world.robot_positions)
    M = len(world.target_positions)

    total_dim = 6 * A + 6 * B + 1
    state = np.zeros(total_dim, dtype=np.float32)

    # 1. 机器人表 (5 * A)
    for i in range(min(N, A)):
        offset = i * 5
        state[offset:offset + 5] = [
            world.robot_positions[i, 0] / L,
            world.robot_positions[i, 1] / L,
            world.robot_velocities[i, 0] / V,
            world.robot_velocities[i, 1] / V,
            world.robot_radii[i] / L,
        ]

    # 2. 机器人存在标记 (A)
    flag_offset = 5 * A
    for i in range(min(N, A)):
        state[flag_offset + i] = 1.0

    # 3. 目标表 (5 * B)
    target_offset = 6 * A
    for j in range(min(M, B)):
        offset = target_offset + j * 5
        state[offset:offset + 5] = [
            world.target_positions[j, 0] / L,
            world.target_positions[j, 1] / L,
            world.target_velocities[j, 0] / V,
            world.target_velocities[j, 1] / V,
            world.target_radii[j] / L,
        ]

    # 4. 目标存在标记 (B)
    target_flag_offset = target_offset + 5 * B
    for j in range(min(M, B)):
        state[target_flag_offset + j] = 1.0

    # 5. 剩余时间比例 (1)
    horizon = config.horizon
    state[-1] = float((horizon - world.step_index) / horizon)

    return np.ascontiguousarray(state, dtype=np.float32)
