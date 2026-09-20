"""任务域（domain）定义：允许的任务超参数取值范围，以及配置对域的校验。

domain 文件为单一 YAML 文档，evaluation_count_pairs 必须是 training_count_pairs 的子集。
"""
from math import isfinite, pi
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
import yaml

from coverage_bench.config import TaskConfig, UniqueSafeLoader
from coverage_bench.errors import ConfigurationError, CoverageError


class TaskDomain(BaseModel):
    """任务域：各超参数允许的取值范围（严格模式，多余字段禁止）。"""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    domain_schema_version: Literal["coverage-domain/1.0"]
    protocol_version: Literal["coverage-policy/1.0"]
    task_version: Literal["coverage-task/1.0"]
    training_count_pairs: tuple[tuple[int, int], ...]
    evaluation_count_pairs: tuple[tuple[int, int], ...]
    horizon_range: tuple[int, int]
    sense_radius_range: tuple[float, float]
    target_max_speed_range: tuple[float, float]
    target_radius_range: tuple[float, float]
    turn_interval_steps_range: tuple[int, int]
    target_speed_fraction_range: tuple[float, float]
    layout_kinds: tuple[str, ...]
    robot_min_gap_range: tuple[float, float]
    crossing_band_fraction_range: tuple[float, float]
    crossing_angle_jitter_range: tuple[float, float]
    cluster_radius_range: tuple[float, float]

    @field_validator("training_count_pairs", mode="before")
    @classmethod
    def _val_train_pairs(cls, v: Any) -> tuple[tuple[int, int], ...]:
        if not isinstance(v, (list, tuple)):
            raise ConfigurationError("training_count_pairs must be list or tuple", code="CONFIG_INVALID")
        out = []
        for pair in v:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ConfigurationError("Each training count pair must have length 2", code="CONFIG_INVALID")
            a, b = pair
            if type(a) is not int or type(b) is not int:
                raise ConfigurationError("Pair values must be int", code="CONFIG_INVALID")
            if a < 1 or a > 8 or b < 1 or b > 8:
                raise ConfigurationError("Pair values must be in [1, 8]", code="CONFIG_INVALID")
            out.append((a, b))
        return tuple(out)

    @field_validator("evaluation_count_pairs", mode="before")
    @classmethod
    def _val_eval_pairs(cls, v: Any) -> tuple[tuple[int, int], ...]:
        if not isinstance(v, (list, tuple)):
            raise ConfigurationError("evaluation_count_pairs must be list or tuple", code="CONFIG_INVALID")
        out = []
        for pair in v:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ConfigurationError("Each evaluation count pair must have length 2", code="CONFIG_INVALID")
            a, b = pair
            if type(a) is not int or type(b) is not int:
                raise ConfigurationError("Pair values must be int", code="CONFIG_INVALID")
            if a < 1 or a > 8 or b < 1 or b > 8:
                raise ConfigurationError("Pair values must be in [1, 8]", code="CONFIG_INVALID")
            out.append((a, b))
        return tuple(out)

    @field_validator("horizon_range", mode="before")
    @classmethod
    def _val_horizon_range(cls, v: Any) -> tuple[int, int]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ConfigurationError("horizon_range must have length 2", code="CONFIG_INVALID")
        for x in v:
            if type(x) is not int:
                raise ConfigurationError("horizon_range values must be int", code="CONFIG_INVALID")
        if v[0] < 1 or v[1] > 256 or v[0] > v[1]:
            raise ConfigurationError("horizon_range out of bounds [1, 256]", code="CONFIG_INVALID")
        return (v[0], v[1])

    @field_validator("sense_radius_range", mode="before")
    @classmethod
    def _val_sense_range(cls, v: Any) -> tuple[float, float]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ConfigurationError("sense_radius_range must have length 2", code="CONFIG_INVALID")
        out = []
        for x in v:
            if type(x) is bool or not isinstance(x, (float, int)) or not isfinite(float(x)):
                raise ConfigurationError("sense_radius_range values must be finite float", code="CONFIG_INVALID")
            out.append(float(x))
        if out[0] < 0.0 or out[0] > out[1]:
            raise ConfigurationError("sense_radius_range out of bounds", code="CONFIG_INVALID")
        return (out[0], out[1])

    @field_validator("target_max_speed_range", mode="before")
    @classmethod
    def _val_speed_range(cls, v: Any) -> tuple[float, float]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ConfigurationError("target_max_speed_range must have length 2", code="CONFIG_INVALID")
        out = []
        for x in v:
            if type(x) is bool or not isinstance(x, (float, int)) or not isfinite(float(x)):
                raise ConfigurationError("target_max_speed_range values must be finite float", code="CONFIG_INVALID")
            out.append(float(x))
        if out[0] < 0.0 or out[0] > out[1]:
            raise ConfigurationError("target_max_speed_range out of bounds", code="CONFIG_INVALID")
        return (out[0], out[1])

    @field_validator("target_radius_range", mode="before")
    @classmethod
    def _val_radius_range(cls, v: Any) -> tuple[float, float]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ConfigurationError("target_radius_range must have length 2", code="CONFIG_INVALID")
        out = []
        for x in v:
            if type(x) is bool or not isinstance(x, (float, int)) or not isfinite(float(x)):
                raise ConfigurationError("target_radius_range values must be finite float", code="CONFIG_INVALID")
            out.append(float(x))
        if out[0] <= 0.0 or out[1] >= 1.0 or out[0] > out[1]:
            raise ConfigurationError("target_radius_range out of bounds (0.0, 1.0)", code="CONFIG_INVALID")
        return (out[0], out[1])

    @field_validator("turn_interval_steps_range", mode="before")
    @classmethod
    def _val_turn_range(cls, v: Any) -> tuple[int, int]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ConfigurationError("turn_interval_steps_range must have length 2", code="CONFIG_INVALID")
        for x in v:
            if type(x) is not int:
                raise ConfigurationError("turn_interval_steps_range values must be int", code="CONFIG_INVALID")
        if v[0] < 1 or v[0] > v[1]:
            raise ConfigurationError("turn_interval_steps_range out of bounds", code="CONFIG_INVALID")
        return (v[0], v[1])

    @field_validator("target_speed_fraction_range", mode="before")
    @classmethod
    def _val_fraction_range(cls, v: Any) -> tuple[float, float]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ConfigurationError("target_speed_fraction_range must have length 2", code="CONFIG_INVALID")
        out = []
        for x in v:
            if type(x) is bool or not isinstance(x, (float, int)) or not isfinite(float(x)):
                raise ConfigurationError("target_speed_fraction_range values must be finite float", code="CONFIG_INVALID")
            out.append(float(x))
        if out[0] < 0.0 or out[1] > 1.0 or out[0] > out[1]:
            raise ConfigurationError("target_speed_fraction_range out of bounds [0.0, 1.0]", code="CONFIG_INVALID")
        return (out[0], out[1])

    @field_validator("layout_kinds", mode="before")
    @classmethod
    def _val_layout_kinds(cls, v: Any) -> tuple[str, ...]:
        if not isinstance(v, (list, tuple)) or len(v) == 0:
            raise ConfigurationError("layout_kinds must be non-empty", code="CONFIG_INVALID")
        allowed = {"uniform", "crossing", "clustered"}
        for k in v:
            if k not in allowed:
                raise ConfigurationError(f"Unknown layout kind {k}", code="CONFIG_INVALID")
        return tuple(v)

    @field_validator("robot_min_gap_range", mode="before")
    @classmethod
    def _val_gap_range(cls, v: Any) -> tuple[float, float]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ConfigurationError("robot_min_gap_range must have length 2", code="CONFIG_INVALID")
        out = []
        for x in v:
            if type(x) is bool or not isinstance(x, (float, int)) or not isfinite(float(x)):
                raise ConfigurationError("robot_min_gap_range values must be finite float", code="CONFIG_INVALID")
            out.append(float(x))
        if out[0] < 0.0 or out[0] > out[1]:
            raise ConfigurationError("robot_min_gap_range out of bounds", code="CONFIG_INVALID")
        return (out[0], out[1])

    @field_validator("crossing_band_fraction_range", mode="before")
    @classmethod
    def _val_band_range(cls, v: Any) -> tuple[float, float]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ConfigurationError("crossing_band_fraction_range must have length 2", code="CONFIG_INVALID")
        out = []
        for x in v:
            if type(x) is bool or not isinstance(x, (float, int)) or not isfinite(float(x)):
                raise ConfigurationError("crossing_band_fraction_range values must be finite float", code="CONFIG_INVALID")
            out.append(float(x))
        if out[0] <= 0.0 or out[1] > 1.0 or out[0] > out[1]:
            raise ConfigurationError("crossing_band_fraction_range out of bounds (0.0, 1.0]", code="CONFIG_INVALID")
        return (out[0], out[1])

    @field_validator("crossing_angle_jitter_range", mode="before")
    @classmethod
    def _val_jitter_range(cls, v: Any) -> tuple[float, float]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ConfigurationError("crossing_angle_jitter_range must have length 2", code="CONFIG_INVALID")
        out = []
        for x in v:
            if type(x) is bool or not isinstance(x, (float, int)) or not isfinite(float(x)):
                raise ConfigurationError("crossing_angle_jitter_range values must be finite float", code="CONFIG_INVALID")
            out.append(float(x))
        if out[0] < 0.0 or out[1] >= pi / 2 or out[0] > out[1]:
            raise ConfigurationError("crossing_angle_jitter_range out of bounds [0.0, pi/2)", code="CONFIG_INVALID")
        return (out[0], out[1])

    @field_validator("cluster_radius_range", mode="before")
    @classmethod
    def _val_cluster_range(cls, v: Any) -> tuple[float, float]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ConfigurationError("cluster_radius_range must have length 2", code="CONFIG_INVALID")
        out = []
        for x in v:
            if type(x) is bool or not isinstance(x, (float, int)) or not isfinite(float(x)):
                raise ConfigurationError("cluster_radius_range values must be finite float", code="CONFIG_INVALID")
            out.append(float(x))
        if out[0] <= 0.0 or out[1] >= 1.0 or out[0] > out[1]:
            raise ConfigurationError("cluster_radius_range out of bounds (0.0, 1.0)", code="CONFIG_INVALID")
        return (out[0], out[1])


def load_task_domain(path: Path) -> TaskDomain:
    """读取并校验 domain YAML 文件，返回 TaskDomain。"""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise ConfigurationError(f"Failed to read domain file: {exc}", code="CONFIG_INVALID")

    if "!!python" in text:
        raise ConfigurationError("Unsafe Python tags are forbidden in YAML", code="CONFIG_INVALID")

    try:
        docs = list(yaml.load_all(text, Loader=UniqueSafeLoader))
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"YAML parsing failed: {exc}", code="CONFIG_INVALID")

    if len(docs) != 1:
        raise ConfigurationError("Domain file must contain exactly one YAML document", code="CONFIG_INVALID")

    data = docs[0]
    if not isinstance(data, dict):
        raise ConfigurationError("Domain YAML document root must be a mapping", code="CONFIG_INVALID")

    try:
        domain = TaskDomain(**data)
    except ConfigurationError:
        raise
    except ValidationError as err:
        first_error = err.errors()[0]
        field = ".".join(str(loc) for loc in first_error.get("loc", []))
        raise ConfigurationError(f"Domain validation error: {first_error.get('msg')}", code="CONFIG_INVALID", field=field)

    # 验证 evaluation_count_pairs 必须是 training_count_pairs 的子集
    training_set = set(domain.training_count_pairs)
    for eval_pair in domain.evaluation_count_pairs:
        if eval_pair not in training_set:
            raise ConfigurationError(f"Evaluation count pair {eval_pair} not in training count pairs", code="CONFIG_INVALID", field="evaluation_count_pairs")

    return domain


def validate_config_in_domain(config: TaskConfig, domain: TaskDomain, mode: str) -> None:
    """校验 TaskConfig 是否落在 domain 允许范围内（mode 为 training 或 evaluation）。"""
    pair = (config.num_agents, config.num_targets)
    if mode == "training":
        if pair not in domain.training_count_pairs:
            raise ConfigurationError(f"Agent/target pair {pair} not allowed in training domain", code="CONFIG_INVALID", field="num_agents")
    elif mode == "evaluation":
        if pair not in domain.evaluation_count_pairs:
            raise ConfigurationError(f"Agent/target pair {pair} not allowed in evaluation domain", code="CONFIG_INVALID", field="num_agents")
    else:
        raise ConfigurationError(f"Unknown domain mode: {mode}", code="CONFIG_INVALID")

    if config.horizon < domain.horizon_range[0] or config.horizon > domain.horizon_range[1]:
        raise ConfigurationError("horizon outside domain range", code="CONFIG_INVALID", field="horizon")

    pub = config.public
    if pub.sense_radius < domain.sense_radius_range[0] or pub.sense_radius > domain.sense_radius_range[1]:
        raise ConfigurationError("sense_radius outside domain range", code="CONFIG_INVALID", field="public.sense_radius")
    if pub.target_max_speed < domain.target_max_speed_range[0] or pub.target_max_speed > domain.target_max_speed_range[1]:
        raise ConfigurationError("target_max_speed outside domain range", code="CONFIG_INVALID", field="public.target_max_speed")
    if pub.target_radius < domain.target_radius_range[0] or pub.target_radius > domain.target_radius_range[1]:
        raise ConfigurationError("target_radius outside domain range", code="CONFIG_INVALID", field="public.target_radius")

    t_min, t_max = pub.turn_interval_steps
    if t_min < domain.turn_interval_steps_range[0] or t_max > domain.turn_interval_steps_range[1]:
        raise ConfigurationError("turn_interval_steps outside domain range", code="CONFIG_INVALID", field="public.turn_interval_steps")

    f_min, f_max = pub.target_speed_fraction
    if f_min < domain.target_speed_fraction_range[0] or f_max > domain.target_speed_fraction_range[1]:
        raise ConfigurationError("target_speed_fraction outside domain range", code="CONFIG_INVALID", field="public.target_speed_fraction")

    scen = config.scenario
    if scen.layout_kind not in domain.layout_kinds:
        raise ConfigurationError("layout_kind not in domain allowed layout kinds", code="CONFIG_INVALID", field="scenario.layout_kind")
    if scen.robot_min_gap < domain.robot_min_gap_range[0] or scen.robot_min_gap > domain.robot_min_gap_range[1]:
        raise ConfigurationError("robot_min_gap outside domain range", code="CONFIG_INVALID", field="scenario.robot_min_gap")
    if scen.crossing_band_fraction < domain.crossing_band_fraction_range[0] or scen.crossing_band_fraction > domain.crossing_band_fraction_range[1]:
        raise ConfigurationError("crossing_band_fraction outside domain range", code="CONFIG_INVALID", field="scenario.crossing_band_fraction")
    if scen.crossing_angle_jitter < domain.crossing_angle_jitter_range[0] or scen.crossing_angle_jitter > domain.crossing_angle_jitter_range[1]:
        raise ConfigurationError("crossing_angle_jitter outside domain range", code="CONFIG_INVALID", field="scenario.crossing_angle_jitter")
    if scen.cluster_radius < domain.cluster_radius_range[0] or scen.cluster_radius > domain.cluster_radius_range[1]:
        raise ConfigurationError("cluster_radius outside domain range", code="CONFIG_INVALID", field="scenario.cluster_radius")
