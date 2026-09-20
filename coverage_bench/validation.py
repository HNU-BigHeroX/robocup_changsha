"""协议对象校验：选手动作、回合上下文与观测结构的完整性与取值范围。"""
import math
from typing import Any

import numpy as np

from coverage_bench.errors import ActionValidationError, CoverageError, ObservationValidationError
from coverage_bench.protocol import EpisodeContext, ProtocolSpec

TOLERANCE = 8.0 * float(np.finfo(np.float32).eps)


def validate_action(action: Any) -> None:
    """校验动作必须是形状 (2,)、取值 [-1, 1] 的有限 float32 数组。"""
    if not isinstance(action, np.ndarray):
        raise ActionValidationError("Action must be a numpy ndarray", field="action")
    if action.dtype != np.float32:
        raise ActionValidationError("Action dtype must be float32", field="action")
    if action.shape != (2,):
        raise ActionValidationError("Action shape must be (2,)", field="action")
    if not np.isfinite(action).all():
        raise ActionValidationError("Action contains non-finite values", field="action")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ActionValidationError("Action values must be within [-1, 1]", field="action")


def validate_episode_context(context: Any, spec: ProtocolSpec) -> None:
    """校验回合上下文的字段类型与取值范围是否符合协议容量上限。"""
    if not isinstance(context, EpisodeContext):
        raise CoverageError("Context must be EpisodeContext", field="context")
    if type(context.agent_index) is not int or context.agent_index < 0 or context.agent_index >= context.num_agents:
        raise CoverageError("agent_index must be an int in [0, num_agents-1]", field="agent_index")
    if type(context.num_agents) is not int or context.num_agents < 1 or context.num_agents > spec.agent_capacity:
        raise CoverageError("num_agents must be an int in [1, agent_capacity]", field="num_agents")
    if type(context.num_targets) is not int or context.num_targets < 1 or context.num_targets > spec.target_capacity:
        raise CoverageError("num_targets must be an int in [1, target_capacity]", field="num_targets")
    if type(context.horizon) is not int or context.horizon < 1 or context.horizon > spec.max_episode_steps:
        raise CoverageError("horizon must be an int in [1, max_episode_steps]", field="horizon")
    if type(context.policy_seed) is not int or context.policy_seed < 0 or context.policy_seed > (1 << 64) - 1:
        raise CoverageError("policy_seed must be an int in [0, 2**64-1]", field="policy_seed")


def validate_observation(observation: Any, spec: ProtocolSpec, episode: EpisodeContext | None = None) -> None:
    """校验观测结构：键集合、dtype/shape、可见性与存在性约束及取值范围。

    提供 episode 时额外校验观测与回合上下文一致（活跃掩码、感知半径、时间归一化）。
    """
    if not isinstance(observation, dict):
        raise ObservationValidationError("Observation must be a dict", field="observation")

    required_keys = {
        "self_state", "peers", "peer_exists", "peer_visible",
        "targets", "target_exists", "target_visible",
        "agent_index", "step_index", "time_remaining",
    }
    obs_keys = set(observation.keys())
    if obs_keys != required_keys:
        missing = required_keys - obs_keys
        extra = obs_keys - required_keys
        msg = f"Observation keys mismatch. Missing: {missing}, Extra: {extra}"
        field_name = next(iter(missing or extra))
        raise ObservationValidationError(msg, field=field_name)

    agent_index = observation["agent_index"]
    if type(agent_index) is not np.int64:
        raise ObservationValidationError("agent_index must be a np.int64 scalar", field="agent_index")
    if agent_index < 0 or agent_index >= spec.agent_capacity:
        raise ObservationValidationError("agent_index out of bounds", field="agent_index")

    step_index = observation["step_index"]
    if type(step_index) is not np.int64:
        raise ObservationValidationError("step_index must be a np.int64 scalar", field="step_index")
    if step_index < 0 or step_index > spec.max_episode_steps:
        raise ObservationValidationError("step_index out of bounds", field="step_index")

    time_remaining = observation["time_remaining"]
    if type(time_remaining) is not np.float32:
        raise ObservationValidationError("time_remaining must be a np.float32 scalar", field="time_remaining")
    if not np.isfinite(time_remaining) or time_remaining < 0.0 or time_remaining > 1.0:
        raise ObservationValidationError("time_remaining out of bounds [0, 1]", field="time_remaining")

    array_fields = {
        "self_state": (np.float32, (5,)),
        "peers": (np.float32, (spec.agent_capacity, 5)),
        "peer_exists": (np.bool_, (spec.agent_capacity,)),
        "peer_visible": (np.bool_, (spec.agent_capacity,)),
        "targets": (np.float32, (spec.target_capacity, 3)),
        "target_exists": (np.bool_, (spec.target_capacity,)),
        "target_visible": (np.bool_, (spec.target_capacity,)),
    }

    for field, (expected_dtype, expected_shape) in array_fields.items():
        arr = observation[field]
        if not isinstance(arr, np.ndarray):
            raise ObservationValidationError(f"{field} must be a numpy ndarray", field=field)
        if arr.dtype != expected_dtype:
            raise ObservationValidationError(f"{field} dtype must be {expected_dtype}", field=field)
        if arr.shape != expected_shape:
            raise ObservationValidationError(f"{field} shape must be {expected_shape}", field=field)
        if not arr.flags.c_contiguous:
            raise ObservationValidationError(f"{field} must be C-contiguous", field=field)
        if expected_dtype == np.float32 and not np.isfinite(arr).all():
            raise ObservationValidationError(f"{field} contains non-finite values", field=field)

    self_state = observation["self_state"]
    peers = observation["peers"]
    peer_exists = observation["peer_exists"]
    peer_visible = observation["peer_visible"]
    targets = observation["targets"]
    target_exists = observation["target_exists"]
    target_visible = observation["target_visible"]

    if np.any(~peer_exists & peer_visible):
        raise ObservationValidationError("peer_visible implies peer_exists", field="peer_visible")
    if np.any(~target_exists & target_visible):
        raise ObservationValidationError("target_visible implies target_exists", field="target_visible")

    self_idx = int(agent_index)
    if peer_exists[self_idx] or peer_visible[self_idx] or np.any(peers[self_idx] != 0):
        raise ObservationValidationError("Self slot in peers must be inactive and zero", field="peers")

    for i in range(spec.agent_capacity):
        if not peer_visible[i] and np.any(peers[i] != 0):
            raise ObservationValidationError(f"Non-visible peer {i} must be all zeros", field="peers")
    for j in range(spec.target_capacity):
        if not target_visible[j] and np.any(targets[j] != 0):
            raise ObservationValidationError(f"Non-visible target {j} must be all zeros", field="targets")

    tol = TOLERANCE
    if self_state[0] < -1.0 - tol or self_state[0] > 1.0 + tol:
        raise ObservationValidationError("self_state x position out of bounds", field="self_state")
    if self_state[1] < -1.0 - tol or self_state[1] > 1.0 + tol:
        raise ObservationValidationError("self_state y position out of bounds", field="self_state")
    if self_state[2] < -1.0 - tol or self_state[2] > 1.0 + tol:
        raise ObservationValidationError("self_state vx velocity out of bounds", field="self_state")
    if self_state[3] < -1.0 - tol or self_state[3] > 1.0 + tol:
        raise ObservationValidationError("self_state vy velocity out of bounds", field="self_state")
    if self_state[4] <= 0.0 + tol or self_state[4] >= 1.0 - tol:
        raise ObservationValidationError("self_state radius out of bounds", field="self_state")

    for i in range(spec.agent_capacity):
        if peer_visible[i]:
            if peers[i, 0] < -2.0 - tol or peers[i, 0] > 2.0 + tol:
                raise ObservationValidationError(f"peer {i} dx out of bounds", field="peers")
            if peers[i, 1] < -2.0 - tol or peers[i, 1] > 2.0 + tol:
                raise ObservationValidationError(f"peer {i} dy out of bounds", field="peers")
            if peers[i, 2] < -2.0 - tol or peers[i, 2] > 2.0 + tol:
                raise ObservationValidationError(f"peer {i} dvx out of bounds", field="peers")
            if peers[i, 3] < -2.0 - tol or peers[i, 3] > 2.0 + tol:
                raise ObservationValidationError(f"peer {i} dvy out of bounds", field="peers")
            if peers[i, 4] <= 0.0 + tol or peers[i, 4] >= 1.0 - tol:
                raise ObservationValidationError(f"peer {i} radius out of bounds", field="peers")

    for j in range(spec.target_capacity):
        if target_visible[j]:
            if targets[j, 0] < -2.0 - tol or targets[j, 0] > 2.0 + tol:
                raise ObservationValidationError(f"target {j} dx out of bounds", field="targets")
            if targets[j, 1] < -2.0 - tol or targets[j, 1] > 2.0 + tol:
                raise ObservationValidationError(f"target {j} dy out of bounds", field="targets")
            if targets[j, 2] <= 0.0 + tol or targets[j, 2] >= 1.0 - tol:
                raise ObservationValidationError(f"target {j} radius out of bounds", field="targets")

    if episode is not None:
        if agent_index != episode.agent_index:
            raise ObservationValidationError("agent_index does not match episode", field="agent_index")
        if step_index > episode.horizon:
            raise ObservationValidationError("step_index exceeds episode horizon", field="step_index")
        expected_time = np.float32((episode.horizon - step_index) / episode.horizon)
        if time_remaining != expected_time:
            raise ObservationValidationError("time_remaining does not match episode calculation", field="time_remaining")

        expected_peer_exists = np.zeros(spec.agent_capacity, dtype=np.bool_)
        expected_peer_exists[:episode.num_agents] = True
        expected_peer_exists[episode.agent_index] = False
        if not np.array_equal(peer_exists, expected_peer_exists):
            raise ObservationValidationError("peer_exists does not match episode active agents", field="peer_exists")

        expected_target_exists = np.zeros(spec.target_capacity, dtype=np.bool_)
        expected_target_exists[:episode.num_targets] = True
        if not np.array_equal(target_exists, expected_target_exists):
            raise ObservationValidationError("target_exists does not match episode active targets", field="target_exists")

        sense_radius_norm = episode.task.sense_radius / spec.position_scale
        for i in range(spec.agent_capacity):
            if peer_visible[i]:
                dist = math.hypot(float(peers[i, 0]), float(peers[i, 1]))
                if dist > sense_radius_norm + tol:
                    raise ObservationValidationError(f"peer {i} visible beyond sense radius", field="peer_visible")
        for j in range(spec.target_capacity):
            if target_visible[j]:
                dist = math.hypot(float(targets[j, 0]), float(targets[j, 1]))
                if dist > sense_radius_norm + tol:
                    raise ObservationValidationError(f"target {j} visible beyond sense radius", field="target_visible")
