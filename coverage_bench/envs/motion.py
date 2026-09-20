"""目标运动模型：周期性随机转向 + 边界反射，位置按反射定律连续移动。"""
from math import cos, pi, sin
from typing import Tuple

import numpy as np

from coverage_bench.envs.types import ScenarioState


def reflect_coordinate(position: float, velocity: float, dt: float, low: float, high: float) -> Tuple[float, float]:
    """单轴上把质点按速度推进 dt，遇 [low, high] 边界做镜面反射，返回新位置与新速度。"""
    if high <= low or dt < 0 or not low <= position <= high:
        raise ValueError("Invalid reflection interval, position, or dt")
    if velocity == 0.0:
        return float(position), 0.0
    remaining = float(dt)
    pos = float(position)
    vel = float(velocity)
    while True:
        wall = high if vel > 0 else low
        time_to_wall = (wall - pos) / vel
        if time_to_wall > remaining:
            return float(pos + vel * remaining), float(vel)
        pos = wall
        vel = -vel
        remaining -= time_to_wall


def advance_targets(state: ScenarioState) -> None:
    """推进所有目标：到转向步时重新抽取速度方向与下次转向间隔，再逐轴反射移动。"""
    pub = state.config.public
    L = pub.map_half_extent
    r_target = pub.target_radius
    low = -(L - r_target)
    high = +(L - r_target)
    dt = pub.dt
    m = state.config.num_targets

    for j in range(m):
        if state.step_index == state.target_next_turn[j]:
            speed_frac = state.target_rng.uniform(pub.target_speed_fraction[0], pub.target_speed_fraction[1])
            speed = speed_frac * pub.target_max_speed
            angle = state.target_rng.uniform(0.0, 2.0 * pi)
            state.target_velocities[j, 0] = speed * cos(angle)
            state.target_velocities[j, 1] = speed * sin(angle)
            interval = state.target_rng.integers(pub.turn_interval_steps[0], pub.turn_interval_steps[1] + 1)
            state.target_next_turn[j] = state.step_index + interval

        pos_x, vel_x = reflect_coordinate(float(state.target_positions[j, 0]), float(state.target_velocities[j, 0]), dt, low, high)
        pos_y, vel_y = reflect_coordinate(float(state.target_positions[j, 1]), float(state.target_velocities[j, 1]), dt, low, high)
        state.target_positions[j, 0] = pos_x
        state.target_positions[j, 1] = pos_y
        state.target_velocities[j, 0] = vel_x
        state.target_velocities[j, 1] = vel_y
