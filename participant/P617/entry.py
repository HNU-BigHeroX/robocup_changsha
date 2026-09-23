import numpy as np

from coverage_bench.protocol import get_protocol_spec
from coverage_bench.spaces import flatten_observation

_PROTOCOL_SPEC = get_protocol_spec()


class NumpyMLPPolicy:
    """SB3 PPO 导出权重的 numpy 前向推理（训练见 train.py，导出见 tools/export_learning_baseline.py）。"""

    def __init__(self, artifact_dir):
        data = np.load(artifact_dir / "policy.npz")
        self._w1, self._b1 = data["W1"], data["b1"]
        self._w2, self._b2 = data["W2"], data["b2"]
        self._w3, self._b3 = data["W3"], data["b3"]
        self._mean = data["obs_mean"]
        self._var = data["obs_var"]
        self._eps = float(data["obs_eps"])
        self._clip = float(data["obs_clip"])

    def reset(self, context):
        pass

    def act(self, observation):
        x = flatten_observation(observation, _PROTOCOL_SPEC).astype(np.float64)
        # 与 VecNormalize.normalize_obs 同序：先归一化再 clip
        x = np.clip((x - self._mean) / np.sqrt(self._var + self._eps), -self._clip, self._clip)
        h = np.tanh(x @ self._w1 + self._b1)
        h = np.tanh(h @ self._w2 + self._b2)
        action = h @ self._w3 + self._b3
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def close(self):
        pass


def build_policy(context):
    return NumpyMLPPolicy(context.artifact_dir)
