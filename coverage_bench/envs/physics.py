"""机器人动力学：优先走 MPE2 世界积分，无 MPE2 时用内置弹性斥力 + 欧拉积分近似。"""
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from coverage_bench.envs.types import ScenarioState


def apply_robot_boundary(
    position: NDArray[np.float64],
    velocity: NDArray[np.float64],
    radius: float,
    half_extent: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """把机器人位置夹回场地边界（限位而非反射），并清零指向界外的速度分量。"""
    limit = half_extent - radius
    new_pos = np.array(position, dtype=np.float64, copy=True)
    new_vel = np.array(velocity, dtype=np.float64, copy=True)
    for axis in (0, 1):
        if new_pos[axis] > limit:
            new_pos[axis] = limit
            if new_vel[axis] > 0.0:
                new_vel[axis] = 0.0
        elif new_pos[axis] < -limit:
            new_pos[axis] = -limit
            if new_vel[axis] < 0.0:
                new_vel[axis] = 0.0
    return new_pos, new_vel


def advance_robots(state: ScenarioState, actions: Dict[str, NDArray[np.float32]]) -> None:
    """按各 agent 动作推进机器人一步，并写回 state 的位置/速度。"""
    pub = state.config.public
    dt = pub.dt
    mass = pub.robot_mass
    damping = pub.damping
    drive_force = pub.drive_force
    contact_force_val = pub.contact_force
    contact_margin = pub.contact_margin
    max_speed = pub.robot_max_speed
    half_extent = pub.map_half_extent
    n = state.config.num_agents

    if state.mpe_world is not None:
        w = state.mpe_world
        p_force_list: list[np.ndarray | float | None] = [None] * n
        for i in range(n):
            agent_key = f"agent_{i}"
            if agent_key in actions:
                act = np.asarray(actions[agent_key], dtype=np.float64)
                p_force_list[i] = drive_force * act
            else:
                p_force_list[i] = np.zeros(2, dtype=np.float64)
            w.agents[i].state.p_pos = np.array(state.robot_positions[i], dtype=np.float64, copy=True)
            w.agents[i].state.p_vel = np.array(state.robot_velocities[i], dtype=np.float64, copy=True)

        p_force_list = w.apply_environment_force(p_force_list)
        w.integrate_state(p_force_list)

        for i in range(n):
            radius = float(state.robot_radii[i])
            pos = np.array(w.agents[i].state.p_pos, dtype=np.float64, copy=True)
            vel = np.array(w.agents[i].state.p_vel, dtype=np.float64, copy=True)
            pos, vel = apply_robot_boundary(pos, vel, radius, half_extent)
            w.agents[i].state.p_pos = pos
            w.agents[i].state.p_vel = vel
            state.robot_positions[i] = pos
            state.robot_velocities[i] = vel
        return

    p_force = np.zeros((n, 2), dtype=np.float64)

    # 驱动控制力
    for i in range(n):
        agent_key = f"agent_{i}"
        if agent_key in actions:
            act = np.asarray(actions[agent_key], dtype=np.float64)
            p_force[i] += drive_force * act

    # 机器人之间弹性接触斥力
    for i in range(n):
        for j in range(i + 1, n):
            delta = state.robot_positions[i] - state.robot_positions[j]
            dist = float(np.linalg.norm(delta))
            dist_min = float(state.robot_radii[i] + state.robot_radii[j])
            if dist == 0.0:
                normal = np.array([1.0, 0.0], dtype=np.float64)
                penetration = np.logaddexp(0.0, dist_min / contact_margin) * contact_margin
                f = contact_force_val * normal * penetration
            else:
                penetration = np.logaddexp(0.0, -(dist - dist_min) / contact_margin) * contact_margin
                f = contact_force_val * (delta / dist) * penetration
            p_force[i] += f
            p_force[j] -= f

    # 位置先于速度的欧拉积分
    for i in range(n):
        radius = float(state.robot_radii[i])
        pos = state.robot_positions[i] + state.robot_velocities[i] * dt
        vel = (1.0 - damping) * state.robot_velocities[i] + (p_force[i] / mass) * dt
        speed = float(np.linalg.norm(vel))
        if speed > max_speed:
            vel = vel * (max_speed / speed)
        pos, vel = apply_robot_boundary(pos, vel, radius, half_extent)
        state.robot_positions[i] = pos
        state.robot_velocities[i] = vel
