"""参赛模板训练脚本（学习基线示例）：SB3 PPO，训练环境与评测同源。

在训练专用 venv 内运行（torch/sb3/supersuit 仅训练 venv 安装，评测镜像不含）：
    .venv-train/Scripts/python.exe train.py --total-steps 1000000 --curve-points 100
关键约束：训练栈不用 supersuit 的 flatten_observation_v0（其对 Discrete 字段做
one-hot 展开，与评测侧特征顺序不一致）；本文件 FlattenCoverageVecEnv 直接调用
coverage_bench.spaces.flatten_observation，训练与评测共用同一 flatten 实现。

实现说明（针对实际安装版本 stable-baselines3 2.9.0 / supersuit 3.11.0 的三处适配，
flatten 语义与 VecNormalize 设置与设计保持一致）：
1. AliveAgentsBridge：CoverageParallelEnv.possible_agents 按容量给出（agent_capacity），
   而每回合仅 config.num_agents 个 agent 在场；supersuit 的 pettingzoo_env_to_vec_env_v1
   要求 reset 后所有 possible_agents 均有观测，故在桥接处收拢 possible_agents。
2. FlattenCoverageVecEnv.seed/reset：supersuit 3.11 的 ConcatVecEnv 没有 seed() 方法，
   且 SB3 2.9 的 learn() 不再调用 env.seed()，种子只能经 reset(seed=...) 传入；
   因此 seed() 暂存种子，下一次 reset() 时经 ConcatVecEnv.reset(seed) 下发
   （ConcatVecEnv 会给第 i 个副本 seed+i，直达 CoverageParallelEnv.reset(seed=...)）。
3. RecordingVecMonitor：SB3 2.9 的 VecMonitor 不保留历史 episode 回报（也无
   get_episode_rewards()），本子类在 done 时从 info["episode"]["r"] 记录完成回报，
   供训练曲线取最近 100 个 episode 的均值。
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import supersuit as ss
from pettingzoo.utils.wrappers import BaseParallelWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import get_protocol_spec
from coverage_bench.spaces import flatten_observation, flattened_observation_size
from coverage_bench.suites import load_suite

_PUBLIC_SUITE = _REPO_ROOT / "configs" / "public-suite-v1.yaml"
TRAINING_SEED_DEFAULT = 7101  # 训练种子序列，与公开评测种子（public-seeds-v1.json）无关


class AliveAgentsBridge(BaseParallelWrapper):
    """把 possible_agents 对齐到场景实际存活的 agent（num_agents 个）。

    CoverageParallelEnv.possible_agents 按容量 agent_capacity 给出，而每回合只有
    config.num_agents 个 agent 在场；supersuit 的 pettingzoo_env_to_vec_env_v1
    要求 reset 后所有 possible_agents 都有观测，故在训练栈内做本桥接包装。
    """

    def __init__(self, env):
        super().__init__(env)
        self.possible_agents = [f"agent_{i}" for i in range(env.config.num_agents)]

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.possible_agents = list(self.env.agents)
        return obs, infos


class FlattenCoverageVecEnv(VecEnvWrapper):
    """把 agent 拆分后的 Dict 观测压平为与评测侧完全一致的 104 维向量。"""

    def __init__(self, venv):
        super().__init__(venv)
        self._spec = get_protocol_spec()
        size = flattened_observation_size(self._spec)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self._pending_seed = None

    @staticmethod
    def _split_batch(obs):
        """把 gymnasium concatenate 的 dict-of-(n,...) 批量观测拆回 n 个单环境 Dict。"""
        num_envs = len(next(iter(obs.values())))
        return [{key: value[i] for key, value in obs.items()} for i in range(num_envs)]

    def seed(self, seed=None):
        """暂存种子，供下一次 reset() 消费（见模块 docstring 适配 2）。"""
        self._pending_seed = seed
        return [seed]

    def reset(self):
        if self._pending_seed is not None:
            # 绕过 SB3VecEnvWrapper（其 reset(seed) 会调用不存在的 ConcatVecEnv.seed），
            # 直接经 ConcatVecEnv.reset(seed=...) 把种子传给各副本的 pettingzoo 环境。
            obs = self.venv.venv.reset(seed=self._pending_seed)[0]
            self._pending_seed = None
        else:
            obs = self.venv.reset()
        # SB3 VecEnv 约定 reset 返回 (num_envs, obs_dim) 的 ndarray（VecNormalize 有断言）
        return np.stack([flatten_observation(item, self._spec) for item in self._split_batch(obs)])

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        stacked = np.stack([flatten_observation(item, self._spec) for item in self._split_batch(obs)])
        return stacked, rewards, dones, infos


class RecordingVecMonitor(VecMonitor):
    """补记完成 episode 回报历史的 VecMonitor（SB3 2.9 的 VecMonitor 无此功能）。"""

    def __init__(self, venv):
        super().__init__(venv)
        self._completed_rewards = []

    def step_wait(self):
        obs, rewards, dones, infos = super().step_wait()
        for i, done in enumerate(dones):
            if done and isinstance(infos[i], dict) and "episode" in infos[i]:
                self._completed_rewards.append(float(infos[i]["episode"]["r"]))
        return obs, rewards, dones, infos

    def get_episode_rewards(self):
        return self._completed_rewards


def build_env_stack(num_vec_envs: int, seed: int):
    """构造到 FlattenCoverageVecEnv 为止的训练栈（不含 VecMonitor/VecNormalize）。"""
    suite = load_suite(_PUBLIC_SUITE)
    case = suite.groups[0].cases[0]
    env = make_training_env(case.task_config)
    env = AliveAgentsBridge(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=0, base_class="stable_baselines3")
    env = FlattenCoverageVecEnv(env)
    env.seed(seed)
    return env


def train(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_env = build_env_stack(args.num_vec_envs, args.seed)
    env = VecNormalize(RecordingVecMonitor(flat_env), norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO(
        "MlpPolicy", env,
        seed=args.seed,
        n_steps=512, batch_size=256, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, verbose=1,
    )
    started = time.time()
    steps_per_iter = max(args.total_steps // args.curve_points, 1)
    curve_rows = []
    while model.num_timesteps < args.total_steps:
        # SB3 2.9 在 reset_num_timesteps=False 时把 learn 的 total_timesteps 视为相对增量
        # （内部再加 num_timesteps），故这里直接传步长本身，等价于每次推进 steps_per_iter 步。
        model.learn(total_timesteps=steps_per_iter, reset_num_timesteps=False)
        ep_rewards = env.get_episode_rewards()
        mean100 = float(np.mean(ep_rewards[-100:])) if ep_rewards else float("nan")
        curve_rows.append((model.num_timesteps, mean100))
        print(f"[curve] steps={model.num_timesteps} mean_ep_reward(last100)={mean100:.4f}")
    wall = time.time() - started

    model.save(str(out_dir / "model"))
    # 归一化统计量直接落 npz（提交源码禁 pickle——审计 AT-AUD-02 硬拒绝 pickle.load）
    np.savez(
        out_dir / "vecnormalize-stats.npz",
        obs_mean=np.asarray(env.obs_rms.mean, dtype=np.float64),
        obs_var=np.asarray(env.obs_rms.var, dtype=np.float64),
        obs_eps=np.array(env.epsilon, dtype=np.float64),
        obs_clip=np.array(env.clip_obs, dtype=np.float64),
    )
    with open(out_dir / "training_curve.csv", "w", encoding="utf-8", newline="") as f:
        f.write("total_steps,mean_ep_reward_last100\n")
        for steps, reward in curve_rows:
            f.write(f"{steps},{reward:.6f}\n")
    meta = {
        "seed": args.seed,
        "total_steps": model.num_timesteps,
        "wall_seconds": round(wall, 1),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "num_vec_envs": args.num_vec_envs,
        "suite": str(_PUBLIC_SUITE.relative_to(_REPO_ROOT)),
        "train_case": "basic-0",
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"训练完成: {model.num_timesteps} 步, {wall:.1f}s → {out_dir}")


def check_export(args):
    """导出一致性检查：同一批环境观测上比较 SB3 原模型与 numpy 导出模型的确定性动作。

    覆盖观测预处理（flatten 同源函数）、归一化、网络前向、动作裁剪与 dtype 转换。
    """
    import torch

    model_dir = Path(args.model_dir)
    model = PPO.load(str(model_dir / "model.zip"), device="cpu")
    norm = np.load(model_dir / "vecnormalize-stats.npz")
    data = np.load(Path(args.npz))

    # 真实环境 rollout 收集未归一化的 104 维观测（与训练路径同栈、同 flatten）
    env = build_env_stack(args.num_vec_envs, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    raw = []
    obs = env.reset()
    while len(raw) < args.num_obs:
        raw.extend(np.asarray(o, dtype=np.float64) for o in obs)
        actions = [rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) for _ in range(len(obs))]
        obs, _, _, _ = env.step(actions)
    raw = np.stack(raw[: args.num_obs])

    # SB3 路径：用训练侧统计量归一化（先除后 clip，与 SB3 VecNormalize.normalize_obs 一致）
    x_sb3 = np.clip(
        (raw - np.asarray(norm["obs_mean"], dtype=np.float64)) / np.sqrt(np.asarray(norm["obs_var"], dtype=np.float64) + float(norm["obs_eps"])),
        -float(norm["obs_clip"]), float(norm["obs_clip"]),
    )
    # numpy 路径：完全用 npz 内导出统计量，验证统计量逐位一致
    x_np = np.clip(
        (raw - data["obs_mean"]) / np.sqrt(data["obs_var"] + data["obs_eps"]),
        -data["obs_clip"], data["obs_clip"],
    )
    assert np.allclose(x_np, x_sb3, rtol=0.0, atol=1e-12), "npz 归一化统计量与训练侧统计量不一致"

    with torch.no_grad():
        sb3_actions, _ = model.predict(x_sb3, deterministic=True)
    sb3_actions = np.asarray(sb3_actions, dtype=np.float64)

    h = np.tanh(x_np @ data["W1"] + data["b1"])
    h = np.tanh(h @ data["W2"] + data["b2"])
    np_actions = np.clip(h @ data["W3"] + data["b3"], -1.0, 1.0)

    max_diff = float(np.max(np.abs(sb3_actions - np_actions)))
    print(f"样本数: {len(x_sb3)}  max|Δa|: {max_diff:.3e}  容差: 1e-5")
    if max_diff > 1e-5:
        raise SystemExit(f"导出一致性检查失败: max|Δa|={max_diff:.3e} > 1e-5")
    print("导出一致性检查通过")


def main():
    parser = argparse.ArgumentParser(description="参赛模板策略训练（SB3 PPO → npz）")
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--curve-points", type=int, default=20)
    parser.add_argument("--num-vec-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED_DEFAULT)
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "outputs" / "learning-training")
    parser.add_argument("--check-export", action="store_true", help="运行导出一致性检查后退出")
    parser.add_argument("--model-dir", type=Path, default=_REPO_ROOT / "outputs" / "learning-training")
    parser.add_argument("--npz", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "policy.npz")
    parser.add_argument("--num-obs", type=int, default=256)
    args = parser.parse_args()
    if args.check_export:
        check_export(args)
        return
    train(args)


if __name__ == "__main__":
    main()
