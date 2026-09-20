"""coverage_bench 公共出口：MPE2 协同覆盖选拔赛评测库的对外 API。

重型评测子模块（环境、评测）走惰性导入，见 __getattr__。
"""
import importlib
from typing import Any

from coverage_bench.archive import freeze_submission
from coverage_bench.archive_types import FreezeReceipt, FreezeRecord, SubmissionIndex
from coverage_bench.config import TaskConfig, load_task_config
from coverage_bench.errors import CoverageError, EnvironmentStateError
from coverage_bench.protocol import (
    AgentObservation,
    BuildContext,
    EpisodeContext,
    Policy,
    ProtocolSpec,
    PublicTaskParams,
    ResourceLimits,
    RewardBreakdown,
    get_protocol_spec,
)
from coverage_bench.release import (
    ReleaseBundle,
    compute_release_bundle_hash,
    load_release,
    load_release_runtime_limits,
    load_release_schedule,
    load_release_scoring,
    validate_release,
)
from coverage_bench.submission import SubmissionManifest, ValidatedSubmission, validate_submission

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "make_training_env": ("coverage_bench.envs.factory", "make_training_env"),
    "CoverageParallelEnv": ("coverage_bench.envs.parallel_env", "CoverageParallelEnv"),
    "evaluate_submission": ("coverage_bench.evaluation", "evaluate_submission"),
    "verify_submission": ("coverage_bench.evaluation", "verify_submission"),
}


def __getattr__(name: str) -> Any:
    """按需导入 _LAZY_EXPORTS 中登记的重型符号，未登记的名字照常报 AttributeError。"""
    if name in _LAZY_EXPORTS:
        mod_name, attr_name = _LAZY_EXPORTS[name]
        mod = importlib.import_module(mod_name)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "__version__",
    "AgentObservation",
    "BuildContext",
    "CoverageError",
    "CoverageParallelEnv",
    "EnvironmentStateError",
    "EpisodeContext",
    "FreezeReceipt",
    "FreezeRecord",
    "Policy",
    "ProtocolSpec",
    "PublicTaskParams",
    "ReleaseBundle",
    "ResourceLimits",
    "RewardBreakdown",
    "SubmissionIndex",
    "SubmissionManifest",
    "TaskConfig",
    "ValidatedSubmission",
    "compute_release_bundle_hash",
    "evaluate_submission",
    "freeze_submission",
    "get_protocol_spec",
    "load_release",
    "load_release_runtime_limits",
    "load_release_schedule",
    "load_release_scoring",
    "load_task_config",
    "make_training_env",
    "validate_release",
    "validate_submission",
    "verify_submission",
]

