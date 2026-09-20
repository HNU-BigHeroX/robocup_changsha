"""场景初始化：布设机器人与目标，并锁定 MPE2 仿真世界。"""
from math import cos, pi, sin, sqrt

import numpy as np

from coverage_bench.config import TaskConfig
from coverage_bench.envs.types import ScenarioState, WorldSnapshot
from coverage_bench.errors import ScenarioGenerationError


def create_scenario(config: TaskConfig, seed: int) -> ScenarioState:
    """按 seed 生成初始场景（机器人/目标位置、目标初速与转向计划），返回 ScenarioState。

    seed 经 SeedSequence.spawn 拆成 4 路独立随机源；按 layout_kind 布设目标，
    采样失败超限或越界时抛 ScenarioGenerationError。
    """
    seq = np.random.SeedSequence(seed)
    children = seq.spawn(4)
    rng_robots = np.random.Generator(np.random.PCG64(children[0]))
    rng_targets = np.random.Generator(np.random.PCG64(children[1]))
    rng_motion = np.random.Generator(np.random.PCG64(children[2]))
    rng_perm = np.random.Generator(np.random.PCG64(children[3]))

    L = config.public.map_half_extent
    r_robot = config.public.robot_radius
    robot_bound = L - r_robot
    min_gap = config.scenario.robot_min_gap
    limit = config.scenario.sampling_attempt_limit
    n = config.num_agents

    robot_positions = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        attempts = 0
        while True:
            if attempts >= limit:
                raise ScenarioGenerationError("Sampling attempts exhausted during robot placement", code="SCENARIO_GENERATION_FAILED")
            attempts += 1
            x = rng_robots.uniform(-robot_bound, robot_bound)
            y = rng_robots.uniform(-robot_bound, robot_bound)
            pos = np.array([x, y], dtype=np.float64)
            conflict = False
            for j in range(i):
                if np.linalg.norm(pos - robot_positions[j]) < r_robot + r_robot + min_gap:
                    conflict = True
                    break
            if not conflict:
                robot_positions[i] = pos
                break

    robot_velocities = np.zeros((n, 2), dtype=np.float64)
    robot_radii = np.full(n, r_robot, dtype=np.float64)

    r_target = config.public.target_radius
    target_bound = L - r_target
    m = config.num_targets
    layout = config.scenario.layout_kind
    target_positions = np.zeros((m, 2), dtype=np.float64)

    crossing_axis = 0
    if layout == "uniform":
        for j in range(m):
            target_positions[j, 0] = rng_targets.uniform(-target_bound, target_bound)
            target_positions[j, 1] = rng_targets.uniform(-target_bound, target_bound)
    elif layout == "crossing":
        band = config.scenario.crossing_band_fraction * target_bound
        crossing_axis = int(rng_targets.integers(0, 2))
        for j in range(m):
            side = j % 2
            if crossing_axis == 0:
                if side == 0:
                    x = rng_targets.uniform(-target_bound, -target_bound + band)
                else:
                    x = rng_targets.uniform(target_bound - band, target_bound)
                y = rng_targets.uniform(-target_bound, target_bound)
            else:
                if side == 0:
                    y = rng_targets.uniform(-target_bound, -target_bound + band)
                else:
                    y = rng_targets.uniform(target_bound - band, target_bound)
                x = rng_targets.uniform(-target_bound, target_bound)
            target_positions[j] = [x, y]
    elif layout == "clustered":
        c_x = rng_targets.uniform(-target_bound, target_bound)
        c_y = rng_targets.uniform(-target_bound, target_bound)
        cluster_r = config.scenario.cluster_radius
        for j in range(m):
            attempts = 0
            while True:
                if attempts >= limit:
                    raise ScenarioGenerationError("Sampling attempts exhausted during clustered target placement", code="SCENARIO_GENERATION_FAILED")
                attempts += 1
                u = rng_targets.uniform(0.0, 1.0)
                theta = rng_targets.uniform(0.0, 2.0 * pi)
                dist = cluster_r * sqrt(u)
                x = c_x + dist * cos(theta)
                y = c_y + dist * sin(theta)
                if abs(x) <= target_bound and abs(y) <= target_bound:
                    target_positions[j] = [x, y]
                    break

    target_velocities = np.zeros((m, 2), dtype=np.float64)
    target_next_turn = np.zeros(m, dtype=np.int64)
    v_min_frac, v_max_frac = config.public.target_speed_fraction
    t_min, t_max = config.public.turn_interval_steps
    jitter_limit = config.scenario.crossing_angle_jitter

    for j in range(m):
        speed_frac = rng_motion.uniform(v_min_frac, v_max_frac)
        speed = speed_frac * config.public.target_max_speed
        if layout == "crossing":
            if crossing_axis == 0:
                base_angle = 0.0 if (j % 2 == 0) else pi
            else:
                base_angle = pi / 2.0 if (j % 2 == 0) else -pi / 2.0
            jitter = rng_motion.uniform(-jitter_limit, jitter_limit)
            angle = base_angle + jitter
        else:
            angle = rng_motion.uniform(0.0, 2.0 * pi)
        target_velocities[j] = [speed * cos(angle), speed * sin(angle)]
        interval = rng_motion.integers(t_min, t_max + 1)
        target_next_turn[j] = interval

    p_robots = rng_perm.permutation(n)
    robot_positions = robot_positions[p_robots]
    robot_velocities = robot_velocities[p_robots]
    robot_radii = robot_radii[p_robots]

    p_targets = rng_perm.permutation(m)
    target_positions = target_positions[p_targets]
    target_velocities = target_velocities[p_targets]
    target_radii = np.full(m, r_target, dtype=np.float64)
    target_next_turn = target_next_turn[p_targets]

    # 初始化锁定 MPE2 World 仿真世界与 Agent 对象
    from mpe2._mpe_utils.core import World, Agent
    w = World()
    pub = config.public
    w.dt = float(pub.dt)
    w.damping = float(pub.damping)
    w.contact_force = float(pub.contact_force)
    w.contact_margin = float(pub.contact_margin)
    w.dim_p = 2
    w.dim_c = 0
    w.dim_color = 3
    w.agents = []
    for i in range(n):
        a = Agent()
        a.name = f"agent_{i}"
        a.collide = True
        a.movable = True
        a.size = float(robot_radii[i])
        a.initial_mass = float(pub.robot_mass)
        a.max_speed = float(pub.robot_max_speed)
        a.state.p_pos = np.array(robot_positions[i], dtype=np.float64, copy=True)
        a.state.p_vel = np.array(robot_velocities[i], dtype=np.float64, copy=True)
        w.agents.append(a)
    mpe_world = w

    return ScenarioState(
        config=config,
        robot_positions=robot_positions,
        robot_velocities=robot_velocities,
        robot_radii=robot_radii,
        target_positions=target_positions,
        target_velocities=target_velocities,
        target_radii=target_radii,
        step_index=0,
        target_next_turn=target_next_turn,
        target_rng=rng_motion,
        mpe_world=mpe_world,
    )


def snapshot(state: ScenarioState) -> WorldSnapshot:
    """复制当前场景状态为只读世界快照。"""
    return WorldSnapshot(
        robot_positions=state.robot_positions.copy(),
        robot_velocities=state.robot_velocities.copy(),
        robot_radii=state.robot_radii.copy(),
        target_positions=state.target_positions.copy(),
        target_velocities=state.target_velocities.copy(),
        target_radii=state.target_radii.copy(),
        step_index=int(state.step_index),
    )
