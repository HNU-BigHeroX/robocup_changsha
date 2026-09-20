"""任务配置 task YAML 的加载与校验：pydantic 模型定义、数值边界检查、与协议规范对齐。"""
from math import isfinite, pi
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
import yaml

from coverage_bench.errors import ConfigurationError, VersionMismatchError
from coverage_bench.protocol import ProtocolSpec


class UniqueSafeLoader(yaml.SafeLoader):
    """拒绝重复键的 YAML SafeLoader。"""


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigurationError(f"Duplicate key detected in YAML: {key}", code="CONFIG_INVALID", field=str(key))
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


class PublicConfig(BaseModel):
    """对选手公开的仿真参数（地图、动力学、边界与目标运动）。"""
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    map_half_extent: float
    dt: float
    robot_radius: float
    robot_mass: float
    drive_force: float
    damping: float
    robot_max_speed: float
    contact_force: float
    contact_margin: float
    target_radius: float
    target_max_speed: float
    sense_radius: float
    motion_kind: Literal["piecewise_heading_reflect"]
    turn_interval_steps: tuple[int, int]
    target_speed_fraction: tuple[float, float]
    robot_boundary: Literal["clamp_outward_velocity"]
    target_boundary: Literal["reflect_cover_inset"]

    @field_validator("turn_interval_steps", mode="before")
    @classmethod
    def _validate_turn_steps(cls, v: Any) -> tuple[int, int]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ConfigurationError("turn_interval_steps must have length 2", code="CONFIG_INVALID", field="public.turn_interval_steps")
        for x in v:
            if type(x) is not int:
                raise ConfigurationError("turn_interval_steps items must be int", code="CONFIG_INVALID", field="public.turn_interval_steps")
        return (v[0], v[1])

    @field_validator("target_speed_fraction", mode="before")
    @classmethod
    def _validate_speed_fraction(cls, v: Any) -> tuple[float, float]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ConfigurationError("target_speed_fraction must have length 2", code="CONFIG_INVALID", field="public.target_speed_fraction")
        out = []
        for x in v:
            if type(x) is bool or not isinstance(x, (float, int)):
                raise ConfigurationError("target_speed_fraction items must be float", code="CONFIG_INVALID", field="public.target_speed_fraction")
            out.append(float(x))
        return (out[0], out[1])


class ScenarioConfig(BaseModel):
    """场景生成参数（布局类型与各类采样参数）。"""
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    layout_kind: Literal["uniform", "crossing", "clustered"]
    robot_min_gap: float
    sampling_attempt_limit: int
    crossing_band_fraction: float
    crossing_angle_jitter: float
    cluster_radius: float


class TaskConfig(BaseModel):
    """单个任务用例的完整配置（版本号、规模、public/scenario 子配置）。"""
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    config_schema_version: str
    protocol_version: str
    task_version: str
    release_state: Literal["development"]
    num_agents: int
    num_targets: int
    horizon: int
    public: PublicConfig
    scenario: ScenarioConfig
    collision_weight: float


def _validate_numeric_bounds(config: TaskConfig) -> None:
    """检查配置中所有整型/浮点字段的数值边界，越界即抛 ConfigurationError。"""
    # 检查基本整型字段
    if type(config.num_agents) is not int or config.num_agents < 1 or config.num_agents > 8:
        raise ConfigurationError("num_agents out of bounds [1, 8]", code="CONFIG_INVALID", field="num_agents")
    if type(config.num_targets) is not int or config.num_targets < 1 or config.num_targets > 8:
        raise ConfigurationError("num_targets out of bounds [1, 8]", code="CONFIG_INVALID", field="num_targets")
    if type(config.horizon) is not int or config.horizon < 1 or config.horizon > 256:
        raise ConfigurationError("horizon out of bounds [1, 256]", code="CONFIG_INVALID", field="horizon")
    if not isfinite(config.collision_weight) or config.collision_weight <= 0.0:
        raise ConfigurationError("collision_weight must be positive and finite", code="CONFIG_INVALID", field="collision_weight")

    # 检查 public 参数
    pub = config.public
    float_checks = [
        ("public.map_half_extent", pub.map_half_extent, pub.map_half_extent <= 0.0),
        ("public.dt", pub.dt, pub.dt <= 0.0),
        ("public.robot_radius", pub.robot_radius, pub.robot_radius <= 0.0 or pub.robot_radius >= 1.0),
        ("public.robot_mass", pub.robot_mass, pub.robot_mass <= 0.0),
        ("public.drive_force", pub.drive_force, pub.drive_force <= 0.0),
        ("public.damping", pub.damping, pub.damping < 0.0 or pub.damping >= 1.0),
        ("public.robot_max_speed", pub.robot_max_speed, pub.robot_max_speed <= 0.0),
        ("public.contact_force", pub.contact_force, pub.contact_force <= 0.0),
        ("public.contact_margin", pub.contact_margin, pub.contact_margin <= 0.0),
        ("public.target_radius", pub.target_radius, pub.target_radius <= 0.0 or pub.target_radius >= 1.0),
        ("public.target_max_speed", pub.target_max_speed, pub.target_max_speed < 0.0),
        ("public.sense_radius", pub.sense_radius, pub.sense_radius < 0.0),
    ]
    for field_name, val, is_invalid in float_checks:
        if not isfinite(val) or is_invalid:
            raise ConfigurationError(f"{field_name} value out of valid bounds", code="CONFIG_INVALID", field=field_name)

    t_min, t_max = pub.turn_interval_steps
    if t_min < 1 or t_min > t_max:
        raise ConfigurationError("turn_interval_steps out of bounds", code="CONFIG_INVALID", field="public.turn_interval_steps")

    f_min, f_max = pub.target_speed_fraction
    if f_min < 0.0 or f_max > 1.0 or f_min > f_max:
        raise ConfigurationError("target_speed_fraction out of bounds", code="CONFIG_INVALID", field="public.target_speed_fraction")

    # 检查 scenario 参数
    scen = config.scenario
    if not isfinite(scen.robot_min_gap) or scen.robot_min_gap < 0.0:
        raise ConfigurationError("robot_min_gap out of bounds", code="CONFIG_INVALID", field="scenario.robot_min_gap")
    if type(scen.sampling_attempt_limit) is not int or scen.sampling_attempt_limit < 1:
        raise ConfigurationError("sampling_attempt_limit out of bounds", code="CONFIG_INVALID", field="scenario.sampling_attempt_limit")
    if not isfinite(scen.crossing_band_fraction) or scen.crossing_band_fraction <= 0.0 or scen.crossing_band_fraction > 1.0:
        raise ConfigurationError("crossing_band_fraction out of bounds", code="CONFIG_INVALID", field="scenario.crossing_band_fraction")
    if not isfinite(scen.crossing_angle_jitter) or scen.crossing_angle_jitter < 0.0 or scen.crossing_angle_jitter >= pi / 2:
        raise ConfigurationError("crossing_angle_jitter out of bounds", code="CONFIG_INVALID", field="scenario.crossing_angle_jitter")
    if not isfinite(scen.cluster_radius) or scen.cluster_radius <= 0.0 or scen.cluster_radius >= 1.0:
        raise ConfigurationError("cluster_radius out of bounds", code="CONFIG_INVALID", field="scenario.cluster_radius")


def load_task_config(path: Path) -> TaskConfig:
    """从 YAML 文件读取并解析任务配置，格式不合法即抛 ConfigurationError。"""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise ConfigurationError(f"Failed to read task config file: {exc}", code="CONFIG_INVALID")

    # 拒绝带有不安全 python 标签的内容
    if "!!python" in text:
        raise ConfigurationError("Unsafe Python tags are forbidden in YAML", code="CONFIG_INVALID")

    try:
        docs = list(yaml.load_all(text, Loader=UniqueSafeLoader))
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"YAML parsing failed: {exc}", code="CONFIG_INVALID")

    if len(docs) != 1:
        raise ConfigurationError("Config file must contain exactly one YAML document", code="CONFIG_INVALID")

    data = docs[0]
    if not isinstance(data, dict):
        raise ConfigurationError("Config YAML document root must be a mapping", code="CONFIG_INVALID")

    try:
        config = TaskConfig(**data)
    except ConfigurationError:
        raise
    except ValidationError as err:
        first_error = err.errors()[0]
        field = ".".join(str(loc) for loc in first_error.get("loc", []))
        raise ConfigurationError(f"Config validation error: {first_error.get('msg')}", code="CONFIG_INVALID", field=field)

    return config


def validate_task_config(config: TaskConfig, spec: ProtocolSpec) -> None:
    """校验配置与协议规范一致（版本、缩放参数、容量上限）及数值边界。"""
    if config.protocol_version != spec.protocol_version:
        raise VersionMismatchError(f"Protocol version mismatch: {config.protocol_version} != {spec.protocol_version}", code="VERSION_MISMATCH", field="protocol_version")
    if config.task_version != spec.task_version:
        raise VersionMismatchError(f"Task version mismatch: {config.task_version} != {spec.task_version}", code="VERSION_MISMATCH", field="task_version")
    if config.public.map_half_extent != spec.position_scale:
        raise ConfigurationError("map_half_extent must equal spec.position_scale", code="CONFIG_INVALID", field="public.map_half_extent")
    if config.public.robot_max_speed != spec.velocity_scale:
        raise ConfigurationError("robot_max_speed must equal spec.velocity_scale", code="CONFIG_INVALID", field="public.robot_max_speed")
    if config.num_agents > spec.agent_capacity:
        raise ConfigurationError("num_agents exceeds agent_capacity", code="CONFIG_INVALID", field="num_agents")
    if config.num_targets > spec.target_capacity:
        raise ConfigurationError("num_targets exceeds target_capacity", code="CONFIG_INVALID", field="num_targets")
    if config.horizon > spec.max_episode_steps:
        raise ConfigurationError("horizon exceeds max_episode_steps", code="CONFIG_INVALID", field="horizon")

    _validate_numeric_bounds(config)
