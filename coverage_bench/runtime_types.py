"""评测运行时的数据结构：运行配置与单回合执行产出。"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from coverage_bench.protocol import ResourceLimits
from coverage_bench.results import EpisodeRecord, ErrorRecord, HardwareProfile


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """评测运行配置：后端类型、资源限额、硬件档案与基础设施重试上限。"""
    backend: Literal["process", "container"]
    limits: ResourceLimits
    hardware_profile: HardwareProfile
    worker_image_digest: str | None = None
    infrastructure_retry_limit: int = 1


@dataclass(frozen=True, slots=True)
class EpisodeRun:
    """单回合执行产出：回合记录、错误列表、worker 日志路径与 act 耗时序列。"""
    record: EpisodeRecord
    errors: tuple[ErrorRecord, ...]
    worker_logs: tuple[Path, ...]
    act_timings_ms: tuple[float, ...]
