"""环境工厂：校验配置并构建评测用并行环境。"""
from typing import Optional

from coverage_bench.config import TaskConfig, validate_task_config
from coverage_bench.envs.parallel_env import CoverageParallelEnv
from coverage_bench.protocol import get_protocol_spec


def make_training_env(config: TaskConfig, render_mode: Optional[str] = None) -> CoverageParallelEnv:
    """加载协议规格、校验 TaskConfig 后构建 CoverageParallelEnv。"""
    spec = get_protocol_spec()
    validate_task_config(config, spec)
    return CoverageParallelEnv(config=config, spec=spec, render_mode=render_mode)
