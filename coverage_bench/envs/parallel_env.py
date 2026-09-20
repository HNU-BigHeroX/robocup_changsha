"""PettingZoo 并行环境：覆盖任务的 reset/step/render 主循环实现。"""
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pettingzoo import ParallelEnv

from coverage_bench.config import TaskConfig
from coverage_bench.envs.motion import advance_targets
from coverage_bench.envs.physics import advance_robots
from coverage_bench.envs.rendering import render_rgb_frame
from coverage_bench.envs.scenario import create_scenario, snapshot
from coverage_bench.envs.types import ScenarioState, WorldSnapshot
from coverage_bench.errors import CoverageError, EnvironmentStateError
from coverage_bench.metrics import compute_step_metrics
from coverage_bench.observations import encode_global_state, observe_agent
from coverage_bench.protocol import ProtocolSpec
from coverage_bench.rewards import compute_reward
from coverage_bench.spaces import make_action_space, make_observation_space
from coverage_bench.validation import validate_action


class CoverageParallelEnv(ParallelEnv):
    """协同覆盖任务的 PettingZoo 并行环境。

    possible_agents 按协议容量 agent_capacity 给出，而每回合 reset 后只有
    config.num_agents 个 agent 在场；截断后 agents 清空，须 reset 才能继续。
    """

    metadata = {"render_modes": ["rgb_array", "human"], "name": "coverage_env_v1"}

    def __init__(self, config: TaskConfig, spec: ProtocolSpec, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config
        self.spec = spec
        if render_mode is not None and render_mode not in ("rgb_array", "human"):
            raise CoverageError(f"不支持的 render_mode: {render_mode}", code="ENV_RENDER_MODE_INVALID")
        self.render_mode = render_mode

        self.possible_agents = [f"agent_{i}" for i in range(spec.agent_capacity)]
        self.agents: List[str] = []

        self._obs_spaces = {agent: make_observation_space(spec) for agent in self.possible_agents}
        self._act_spaces = {agent: make_action_space() for agent in self.possible_agents}
        state_dim = 6 * spec.agent_capacity + 6 * spec.target_capacity + 1
        self.state_space = make_action_space().__class__(
            -np.inf, np.inf, shape=(state_dim,), dtype=np.float32
        )

        self._scenario_state: Optional[ScenarioState] = None
        self._current_snapshot: Optional[WorldSnapshot] = None
        self._closed: bool = False
        self._viewer: Any | None = None

    def observation_space(self, agent: str):
        return self._obs_spaces[agent]

    def action_space(self, agent: str):
        return self._act_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """按 seed 初始化场景，返回在场 agent 的观测与初始 info。"""
        if self._closed:
            raise EnvironmentStateError("Environment is closed", code="ENV_STATE_INVALID")

        if options is not None:
            if not isinstance(options, dict):
                raise CoverageError("options 必须为字典类型", code="ENV_OPTIONS_INVALID")
            if options:
                raise CoverageError(
                    f"环境契约仅允许 options 为 None 或空字典，收到非空选项: {sorted(options.keys())}",
                    code="ENV_OPTIONS_UNSUPPORTED",
                )

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
                raise CoverageError(f"seed 必须为 64 位非负整数，收到非法类型: {type(seed)}", code="ENV_SEED_INVALID")
            if seed < 0 or seed >= 2**64:
                raise CoverageError(f"seed 超出 64 位非负整数范围: {seed}", code="ENV_SEED_INVALID")
            seed_val = int(seed)
        else:
            seed_val = int.from_bytes(os.urandom(8), "big") & ((1 << 63) - 1)

        self._scenario_state = create_scenario(self.config, seed_val)
        self._current_snapshot = snapshot(self._scenario_state)
        self.agents = [f"agent_{i}" for i in range(self.config.num_agents)]

        obs = {
            agent: observe_agent(self._current_snapshot, i, self.config, self.spec)
            for i, agent in enumerate(self.agents)
        }
        infos = {
            agent: {
                "step_index": 0,
                "metrics": None,
                "reward_terms": None,
            }
            for agent in self.agents
        }
        return obs, infos

    def state(self) -> np.ndarray:
        """返回全局状态向量（须先 reset）。"""
        if self._closed:
            raise EnvironmentStateError("Environment is closed", code="ENV_STATE_INVALID")
        if self._scenario_state is None or self._current_snapshot is None:
            raise EnvironmentStateError("Environment must be reset before querying state", code="ENV_STATE_INVALID")
        return np.copy(encode_global_state(self._current_snapshot, self.config, self.spec))

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """推进仿真一步，返回团队共享奖励的观测/奖励/终止/截断/info 五元组。"""
        if self._closed:
            raise EnvironmentStateError("Environment is closed", code="ENV_STATE_INVALID")
        if self._scenario_state is None or self._current_snapshot is None:
            raise EnvironmentStateError("Environment must be reset before step", code="ENV_STATE_INVALID")

        if len(self.agents) == 0:
            if len(actions) == 0:
                return {}, {}, {}, {}, {}
            raise EnvironmentStateError("Cannot step after truncation without reset", code="ENV_STATE_INVALID")

        if set(actions.keys()) != set(self.agents):
            raise CoverageError(
                f"Action keys {set(actions.keys())} do not match active agents {set(self.agents)}",
                code="ENV_STATE_INVALID",
                field="actions",
            )

        for agent in self.agents:
            validate_action(actions[agent])

        advance_robots(self._scenario_state, actions)
        advance_targets(self._scenario_state)
        self._scenario_state.step_index += 1
        self._current_snapshot = snapshot(self._scenario_state)

        metrics = compute_step_metrics(self._current_snapshot)
        rew_dict = compute_reward(metrics, self.config.collision_weight)
        team_reward = float(rew_dict["team_reward"])

        is_truncated = bool(self._scenario_state.step_index >= self.config.horizon)
        active_agents = list(self.agents)
        if is_truncated:
            self.agents = []

        obs = {
            agent: observe_agent(self._current_snapshot, i, self.config, self.spec)
            for i, agent in enumerate(active_agents)
        }
        rewards = {agent: team_reward for agent in active_agents}
        terminations = {agent: False for agent in active_agents}
        truncations = {agent: is_truncated for agent in active_agents}
        infos = {
            agent: {
                "step_index": self._scenario_state.step_index,
                "metrics": metrics,
                "reward_terms": {
                    "coverage": float(rew_dict["coverage"]),
                    "collision": float(rew_dict["collision"]),
                    "team_reward": team_reward,
                },
            }
            for agent in active_agents
        }

        return obs, rewards, terminations, truncations, infos

    def render(self) -> Optional[np.ndarray]:
        """按 render_mode 渲染当前帧：rgb_array 返回图像，human 弹出 pygame 窗口。"""
        if self._closed:
            raise EnvironmentStateError("Environment is closed", code="ENV_STATE_INVALID")
        if self._current_snapshot is None:
            raise EnvironmentStateError("Environment must be reset before render", code="ENV_STATE_INVALID")
        if self.render_mode == "rgb_array":
            return render_rgb_frame(self._current_snapshot, self.config)
        if self.render_mode == "human":
            img = render_rgb_frame(self._current_snapshot, self.config)
            if self._viewer is None:
                import pygame
                pygame.init()
                pygame.display.init()
                self._viewer = pygame.display.set_mode((img.shape[1], img.shape[0]))
                pygame.display.set_caption("Coverage Bench")
            import pygame
            surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
            self._viewer.blit(surf, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            return None
        return None

    def close(self) -> None:
        self._closed = True
        self._scenario_state = None
        self._current_snapshot = None
        self.agents = []
        if self._viewer is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._viewer = None
