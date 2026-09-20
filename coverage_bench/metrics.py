"""步级指标计算：覆盖率（二分图最大匹配）与碰撞统计。"""
from math import dist

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from coverage_bench.envs.types import StepMetrics, WorldSnapshot


def compute_step_metrics(world: WorldSnapshot) -> StepMetrics:
    """由世界快照计算一步的匹配数、碰撞数、覆盖率/碰撞率与全覆盖标志。"""
    N = len(world.robot_positions)
    M = len(world.target_positions)

    if N == 0 or M == 0:
        matched = 0
    else:
        adj = np.zeros((N, M), dtype=np.int32)
        for i in range(N):
            for j in range(M):
                if dist(world.robot_positions[i], world.target_positions[j]) <= world.target_radii[j]:
                    adj[i, j] = 1
        matching = maximum_bipartite_matching(csr_matrix(adj), perm_type="column")
        matched = int(np.sum(matching >= 0))

    participants = set()
    pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            if dist(world.robot_positions[i], world.robot_positions[j]) < world.robot_radii[i] + world.robot_radii[j]:
                participants.add(i)
                participants.add(j)
                pairs += 1

    collision_agents = len(participants)
    coverage_rate = float(matched / M) if M > 0 else 0.0
    collision_rate = float(collision_agents / N) if N > 0 else 0.0
    full_coverage = bool(matched == M)

    return StepMetrics(
        matched_targets=int(matched),
        collision_agents=int(collision_agents),
        collision_pairs=int(pairs),
        coverage_rate=coverage_rate,
        collision_rate=collision_rate,
        full_coverage=full_coverage,
    )
