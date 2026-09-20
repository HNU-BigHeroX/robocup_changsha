"""评测结果数据模型与读写：回合记录 CSV、result.json、公开摘要与完整性校验。"""
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

from coverage_bench.errors import CoverageError
from coverage_bench.protocol import ResourceLimits

EPISODE_FIELDS = (
    "result_schema_version", "run_id", "participant_id", "group_id", "case_id",
    "repeat_index", "attempt_index", "accepted", "status", "num_agents",
    "num_targets", "horizon", "steps_completed", "return_sum", "mean_j",
    "mean_coverage_rate", "mean_collision_rate", "full_coverage_fraction",
    "collision_pairs_sum", "act_count", "act_ms_sum", "act_ms_p95", "act_ms_max",
    "initialization_ms_sum", "wall_ms", "error_code",
)


class EpisodeRecord(BaseModel):
    """单个回合的完整记录（指标、耗时、状态），附带强完整性校验。"""
    model_config = ConfigDict(extra="forbid")

    result_schema_version: str = "coverage-result/1.0"
    run_id: str
    participant_id: str
    group_id: str
    case_id: str
    repeat_index: int
    attempt_index: int
    accepted: bool
    status: str
    num_agents: int
    num_targets: int
    horizon: int
    steps_completed: int
    return_sum: float | None = None
    mean_j: float | None = None
    mean_coverage_rate: float | None = None
    mean_collision_rate: float | None = None
    full_coverage_fraction: float | None = None
    collision_pairs_sum: int | None = None
    act_count: int
    act_ms_sum: float
    act_ms_p95: float
    act_ms_max: float
    initialization_ms_sum: float
    wall_ms: float
    error_code: str | None = None

    @model_validator(mode="after")
    def validate_episode_record_integrity(self) -> "EpisodeRecord":
        import math
        # 校验所有浮点字段为有限值
        for field_name in (
            "return_sum", "mean_j", "mean_coverage_rate", "mean_collision_rate",
            "full_coverage_fraction", "act_ms_sum", "act_ms_p95", "act_ms_max",
            "initialization_ms_sum", "wall_ms",
        ):
            v = getattr(self, field_name)
            if v is not None and not math.isfinite(v):
                from coverage_bench.errors import CoverageError
                raise CoverageError(f"{field_name} 必须为有限数值: {v}", code="METRIC_NOT_FINITE")

        if self.status == "ok":
            from coverage_bench.errors import CoverageError
            if self.steps_completed != self.horizon:
                raise CoverageError(
                    f"成功回合 steps_completed 必须等于 horizon: {self.steps_completed} != {self.horizon}",
                    code="STEPS_COMPLETED_MISMATCH",
                )
            if self.act_count != self.num_agents * self.horizon:
                raise CoverageError(
                    f"成功回合 act_count 必须等于 num_agents * horizon: {self.act_count} != {self.num_agents * self.horizon}",
                    code="ACT_COUNT_MISMATCH",
                )
            if self.return_sum is None or self.mean_j is None:
                raise CoverageError("成功回合 return_sum 与 mean_j 不能为空", code="METRIC_MISSING")
            if not math.isclose(self.return_sum / self.horizon, self.mean_j, rel_tol=1e-4, abs_tol=1e-4):
                raise CoverageError(
                    f"成功回合 return_sum / horizon 必须与 mean_j 一致: {self.return_sum / self.horizon} != {self.mean_j}",
                    code="RETURN_MEAN_MISMATCH",
                )
            for rate_name in ("mean_coverage_rate", "mean_collision_rate", "full_coverage_fraction"):
                rate_val = getattr(self, rate_name)
                if rate_val is None or not (-1e-6 <= rate_val <= 1.0 + 1e-6):
                    raise CoverageError(f"成功回合 {rate_name} 必须属于 [0, 1]: {rate_val}", code="RATE_OUT_OF_BOUNDS")
            if self.collision_pairs_sum is None or self.collision_pairs_sum < 0:
                raise CoverageError(f"成功回合 collision_pairs_sum 必须为非负整数: {self.collision_pairs_sum}", code="COLLISION_INVALID")
        else:
            if self.accepted:
                from coverage_bench.errors import CoverageError
                raise CoverageError("失败回合 accepted 必须为 False", code="FAILED_ACCEPTED_INVALID")
        return self


class GroupMetrics(BaseModel):
    """一个评分组聚合后的指标（场景均值、bootstrap CI、动作耗时统计）。"""
    model_config = ConfigDict(extra="forbid")

    num_scenarios: int
    num_repeats: int
    num_episodes: int
    mean_return: float
    mean_j: float
    mean_coverage_rate: float
    mean_collision_rate: float
    full_coverage_fraction: float
    ci95_j: tuple[float, float] | None = None
    act_ms_mean: float
    act_ms_p95: float
    act_ms_max: float


class HardwareProfile(BaseModel):
    """评测运行的硬件环境档案（CPU、内存、镜像与依赖锁哈希）。"""
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    os_image: str
    cpu_model: str
    cpu_cores_allocated: int
    memory_mib: int
    python_version: str
    official_lock_hash: str
    worker_image_digest: str | None = None


class ErrorRecord(BaseModel):
    """单条评测错误记录（错误码、阶段、归属方与上下文定位）。"""
    model_config = ConfigDict(extra="forbid")

    code: str
    stage: str
    owner: str
    case_id: str | None = None
    repeat_index: int | None = None
    agent_index: int | None = None
    step_index: int | None = None
    field: str | None = None
    message: str
    retryable: bool = False


class EvaluationResult(BaseModel):
    """一次完整评测的结果（溯源哈希、组指标、performance_score 与产物哈希）。"""
    model_config = ConfigDict(extra="forbid")

    result_schema_version: str = "coverage-result/1.0"
    run_id: str
    participant_id: str
    status: str
    provenance: str
    source_commit: str | None = None
    working_tree_dirty: bool = False
    code_tree_hash: str
    inference_lock_hash: str
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    official_commit: str | None = None
    release_bundle_hash: str
    release_state: str
    suite_visibility: str
    protocol_version: str
    task_version: str
    score_version: str
    suite_id: str
    suite_hash: str
    seed_schedule_id: str
    hardware_profile: HardwareProfile | dict[str, Any]
    runtime_limits: ResourceLimits | dict[str, Any]
    group_metrics: dict[str, GroupMetrics]
    performance_score: float | None = None
    performance_score_ci95: tuple[float, float] | list[float] | None = None
    episode_records_path: str = "episodes.csv"
    episode_records_hash: str
    timings_path: str = "timings.npz"
    timings_hash: str
    errors: list[ErrorRecord] = Field(default_factory=list)
    started_at: str
    finished_at: str
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_result_invariants(self) -> "EvaluationResult":
        if self.status != "ok" and self.performance_score is not None:
            raise CoverageError(f"失败结果不允许包含 performance_score: {self.status}")
        if self.performance_score is not None:
            if not math.isfinite(self.performance_score):
                raise CoverageError(f"performance_score 必须为有限数值: {self.performance_score}")
        return self


class PublicScoreSummary(BaseModel):
    """对外公开的分数摘要（剥离私人溯源字段如 inference_lock_hash）。"""
    model_config = ConfigDict(extra="forbid")

    summary_schema_version: str = "coverage-public-summary/1.0"
    participant_id: str
    provenance: str
    status: str
    release_state: str
    suite_visibility: str
    source_commit: str | None = None
    working_tree_dirty: bool = False
    code_tree_hash: str
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    protocol_version: str
    task_version: str
    score_version: str
    suite_id: str
    suite_hash: str
    group_metrics: dict[str, GroupMetrics]
    performance_score: float | None = None
    performance_score_ci95: tuple[float, float] | list[float] | None = None
    wall_ms: float
    generated_at: str


def write_episode_records(records: tuple[EpisodeRecord, ...], path: Path) -> None:
    """将回合记录按固定字段顺序写入 CSV。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(list(EPISODE_FIELDS))
        for record in records:
            row = []
            for field in EPISODE_FIELDS:
                val = getattr(record, field)
                if val is None:
                    row.append("")
                elif isinstance(val, bool):
                    row.append("True" if val else "False")
                elif isinstance(val, float):
                    row.append(repr(val))
                else:
                    row.append(str(val))
            writer.writerow(row)


def read_episode_records(path: Path) -> tuple[EpisodeRecord, ...]:
    """从 CSV 读回回合记录，表头与字段值类型不符即抛 CoverageError。"""
    target = Path(path)
    if not target.exists():
        raise CoverageError(f"回合记录文件不存在: {path}")
    with target.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            raise CoverageError("CSV 文件为空")
        if header != list(EPISODE_FIELDS):
            raise CoverageError(f"CSV 表头不符合规范: {header}")
        
        records = []
        for row_index, row in enumerate(reader, start=1):
            if len(row) != len(EPISODE_FIELDS):
                raise CoverageError(f"第 {row_index} 行字段数量不匹配: 期望 {len(EPISODE_FIELDS)}, 实际 {len(row)}")
            data: dict[str, Any] = {}
            for field, val in zip(EPISODE_FIELDS, row, strict=True):
                if field in ("result_schema_version", "run_id", "participant_id", "group_id", "case_id", "status"):
                    data[field] = val
                elif field in ("repeat_index", "attempt_index", "num_agents", "num_targets", "horizon", "steps_completed", "act_count"):
                    data[field] = int(val)
                elif field == "collision_pairs_sum":
                    data[field] = int(val) if val != "" else None
                elif field == "accepted":
                    if val in ("True", "true", "1"):
                        data[field] = True
                    elif val in ("False", "false", "0"):
                        data[field] = False
                    else:
                        raise CoverageError(f"非法布尔值: {val}")
                elif field == "error_code":
                    data[field] = val if val != "" else None
                else:
                    # 浮点数字段
                    data[field] = float(val) if val != "" else None
            records.append(EpisodeRecord(**data))
    return tuple(records)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_result(result: EvaluationResult, output_dir: Path) -> None:
    """校验产物哈希后原子写入 result.json、errors.jsonl 与 report.md（禁止覆盖既有结果）。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已存在结果文件，避免覆盖既有报告
    result_target = out / "result.json"
    if result_target.exists():
        raise CoverageError(f"输出目录已包含既有评测结果文件，禁止覆盖: {result_target}")

    episodes_file = out / result.episode_records_path
    if not episodes_file.is_file():
        raise CoverageError(f"引用的回合记录文件不存在: {episodes_file}")
    if _file_sha256(episodes_file) != result.episode_records_hash:
        raise CoverageError("回合记录文件哈希不匹配")
        
    timings_file = out / result.timings_path
    if not timings_file.is_file():
        raise CoverageError(f"引用的耗时文件不存在: {timings_file}")
    if _file_sha256(timings_file) != result.timings_hash:
        raise CoverageError("耗时文件哈希不匹配")

    # 在临时位置生成文件并原子移动发布
    tmp_dir = out / ".work"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    tmp_errors = tmp_dir / "errors.tmp.jsonl"
    with tmp_errors.open("w", encoding="utf-8") as stream:
        for err in result.errors:
            stream.write(err.model_dump_json() + "\n")

    tmp_report = tmp_dir / "report.tmp.md"
    md_content = (
        f"# Evaluation Report\n\n"
        f"- Participant: {result.participant_id}\n"
        f"- Status: {result.status}\n"
        f"- Performance Score: {result.performance_score}\n"
        f"- Suite ID: {result.suite_id}\n"
        f"- Started At: {result.started_at}\n"
        f"- Finished At: {result.finished_at}\n"
    )
    tmp_report.write_text(md_content, encoding="utf-8")

    tmp_result = tmp_dir / "result.tmp.json"
    tmp_result.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    # 原子发布
    tmp_errors.replace(out / "errors.jsonl")
    tmp_report.replace(out / "report.md")
    tmp_result.replace(result_target)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)


def read_result(path: Path) -> EvaluationResult:
    """从 result.json 读回评测结果。"""
    target = Path(path)
    if not target.is_file():
        raise CoverageError(f"结果文件不存在: {path}")
    raw_text = target.read_text(encoding="utf-8")
    parsed = json.loads(raw_text)
    return EvaluationResult(**parsed)


def export_public_summary(result: EvaluationResult) -> PublicScoreSummary:
    """由完整结果导出可对外公开的分数摘要。"""
    wall_ms = 0.0
    try:
        start = datetime.fromisoformat(result.started_at)
        finish = datetime.fromisoformat(result.finished_at)
        wall_ms = max(0.0, (finish - start).total_seconds() * 1000.0)
    except Exception:
        wall_ms = 0.0

    return PublicScoreSummary(
        summary_schema_version="coverage-public-summary/1.0",
        participant_id=result.participant_id,
        provenance=result.provenance,
        status=result.status,
        release_state=result.release_state,
        suite_visibility=result.suite_visibility,
        source_commit=result.source_commit,
        working_tree_dirty=result.working_tree_dirty,
        code_tree_hash=result.code_tree_hash,
        artifact_hashes=dict(result.artifact_hashes),
        protocol_version=result.protocol_version,
        task_version=result.task_version,
        score_version=result.score_version,
        suite_id=result.suite_id,
        suite_hash=result.suite_hash,
        group_metrics=dict(result.group_metrics),
        performance_score=result.performance_score,
        performance_score_ci95=result.performance_score_ci95,
        wall_ms=wall_ms,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
