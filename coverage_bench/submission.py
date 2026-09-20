"""提交静态校验：检查目录结构、清单与产物一致性，并生成隔离快照供评测使用。"""
import ast
import hashlib
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from coverage_bench.errors import CoverageError
from coverage_bench.protocol import ResourceLimits
from coverage_bench.release import ReleaseBundle

REQUIRED_SUBMISSION_FILES = (
    "entry.py", "submission.yaml", "artifacts", "requirements-infer.lock",
    "LOG.md", "experiments.csv", "REPORT.md", "THIRD_PARTY.md", "LICENSE",
)

HEX64_REGEX = re.compile(r"^[0-9a-f]{64}$")
PARTICIPANT_ID_REGEX = re.compile(r"^P\d{3}$")


class ArtifactEntry(BaseModel):
    """产物清单条目：artifacts 下的相对路径 + SHA-256 + 字节数。"""
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    size_bytes: int

    @field_validator("size_bytes", mode="before")
    @classmethod
    def validate_size_type(cls, v: Any) -> int:
        if type(v) is not int or isinstance(v, bool):
            raise CoverageError("size_bytes 必须为严格非负整数")
        if v < 0:
            raise CoverageError("size_bytes 不能为负数")
        return v

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, v: str) -> str:
        if not isinstance(v, str) or not HEX64_REGEX.match(v):
            raise CoverageError(f"sha256 必须为 64 位小写十六进制字符串: {v}")
        return v

    @field_validator("path")
    @classmethod
    def validate_path_syntax(cls, v: str) -> str:
        if not isinstance(v, str):
            raise CoverageError("产物路径必须为字符串")
        if v.startswith("/") or v.startswith("\\") or ":" in v:
            raise CoverageError(f"产物路径禁止使用绝对路径或盘符: {v}")
        
        normalized = v.replace("\\", "/")
        parts = normalized.split("/")
        if ".." in parts or "." in parts:
            raise CoverageError(f"产物路径禁止使用相对越界符号: {v}")
        if parts[0] != "artifacts" or len(parts) < 2:
            raise CoverageError(f"产物路径必须位于 artifacts 目录下: {v}")
        return v


class SubmissionManifest(BaseModel):
    """submission.yaml 的严格模型：协议/任务版本、入口与产物登记，多写字段直接拒绝。"""
    model_config = ConfigDict(extra="forbid")

    submission_schema_version: Literal["coverage-submission/1.0"] = "coverage-submission/1.0"
    participant_id: str
    protocol_version: str
    task_version: str
    entrypoint: Literal["entry:build_policy"] = "entry:build_policy"
    artifact_dir: Literal["artifacts"] = "artifacts"
    checkpoint_manifest: list[ArtifactEntry]
    inference_lock: Literal["requirements-infer.lock"] = "requirements-infer.lock"
    method_type: Literal["rule", "learning", "hybrid"]
    train_command: str | None = None
    training_config_paths: list[str] = Field(default_factory=list)
    report_path: Literal["REPORT.md"] = "REPORT.md"
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("participant_id")
    @classmethod
    def validate_participant_id(cls, v: str) -> str:
        if not isinstance(v, str) or not PARTICIPANT_ID_REGEX.match(v):
            raise CoverageError(f"参赛者编号必须符合 Pxxx 规则: {v}")
        return v

    @field_validator("train_command", mode="before")
    @classmethod
    def validate_train_command_type(cls, v: Any) -> str | None:
        if v is not None and not isinstance(v, str):
            raise CoverageError("train_command 必须为字符串或 null")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, str]:
        if len(v) > 16:
            raise CoverageError("metadata 数量不能超过 16 项")
        res: dict[str, str] = {}
        for key, val in v.items():
            if not isinstance(key, str) or len(key) > 64:
                raise CoverageError(f"metadata 键必须为不超过 64 字符的字符串: {key}")
            if not isinstance(val, str) or len(val) > 512:
                raise CoverageError(f"metadata 值必须为不超过 512 字符的字符串: {val}")
            res[key] = val
        return res

    @model_validator(mode="after")
    def validate_training_command_requirement(self) -> "SubmissionManifest":
        if self.method_type in ("learning", "hybrid"):
            if not self.train_command or not self.train_command.strip():
                raise CoverageError(f"{self.method_type} 方法必须提供非空 train_command")
        return self


@dataclass(frozen=True, slots=True)
class ValidatedSubmission:
    """校验通过的提交：清单 + 各项哈希，root 指向隔离快照而非原始目录。"""
    manifest: SubmissionManifest
    root: Path
    manifest_path: Path
    artifact_hashes: dict[str, str]
    inference_lock_hash: str
    code_tree_hash: str
    manifest_sha256: str
    source_commit: str | None
    working_tree_dirty: bool
    snapshot_id: str
    protocol_version: str
    task_version: str


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def working_tree_dirty_status(git_dir: Path) -> bool:
    """返回工作区是否存在未提交内容，被忽略的内容同样计入。

    被忽略的文件不会被 `git status --porcelain` 报告。冻结记录的提交身份必须覆盖
    记录的代码树、模型与清单，因此被忽略的内容同样属于未提交内容。
    """
    status_proc = subprocess.run(
        ["git", "status", "--porcelain", "--ignored", "--", "."],
        cwd=str(git_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if status_proc.returncode != 0:
        raise CoverageError(
            f"无法确定 Git 工作区状态: {status_proc.stderr.strip()}", code="GIT_STATUS_UNAVAILABLE",
        )
    return bool(status_proc.stdout.strip())


def _check_protected_dependencies(lock_path: Path, protected: dict[str, str]) -> None:
    """检查依赖锁内未通过改名（大小写/下划线归一）替换受保护依赖。"""
    text = lock_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkg_part = line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].split(";")[0].split()[0].strip()
        pkg_norm = pkg_part.lower().replace("_", "-")
        for prot_pkg in protected:
            prot_norm = prot_pkg.lower().replace("_", "-")
            if pkg_norm == prot_norm:
                raise CoverageError(
                    f"依赖锁文件中尝试替换受保护依赖: {pkg_part}",
                    code="DEPENDENCY_CONFLICT",
                )


def _check_entrypoint_ast(entry_file: Path) -> None:
    """静态检查 entry.py 暴露了 build_policy 函数（或同名赋值）。"""
    try:
        tree = ast.parse(entry_file.read_text(encoding="utf-8"), filename=str(entry_file))
    except Exception as exc:
        raise CoverageError(f"解析 entry.py 语法树失败: {exc}")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build_policy":
            return
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "build_policy":
                    return
    raise CoverageError("entry.py 未暴露 build_policy 入口")


def validate_submission(
    manifest_path: Path,
    release: ReleaseBundle | None = None,
    limits: ResourceLimits | None = None,
) -> ValidatedSubmission:
    """完整静态校验：必需文件、清单、入口、依赖与产物哈希，通过后返回隔离快照视图。"""
    if release is None:
        from coverage_bench.protocol import get_protocol_spec
        release = ReleaseBundle(
            release_schema_version="coverage-release/1.0",
            release_id="development",
            release_state="development",
            official_commit=None,
            protocol=get_protocol_spec(),
            score_version="coverage-score/1.0",
            components=(),
            protected_dependencies={"numpy": ">=1.26.0"},
            mpe2_source_commit=None,
            mpe2_distribution_sha256=None,
            created_at="2026-09-18T00:00:00+00:00",
        )
    if limits is None:
        limits = ResourceLimits()

    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise CoverageError(f"提交清单文件不存在: {manifest_path}")

    sub_dir = manifest_file.parent
    if sub_dir.name != manifest_file.stem and not PARTICIPANT_ID_REGEX.match(sub_dir.name):
        pass

    # 查询原始目录真实 Git 提交与干净状态
    source_commit: Optional[str] = None
    working_tree_dirty: bool = True
    try:
        commit_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(sub_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_proc.returncode == 0 and commit_proc.stdout.strip():
            source_commit = commit_proc.stdout.strip()
            working_tree_dirty = working_tree_dirty_status(sub_dir)
    except Exception:
        pass

    # 1. 检查必需文件是否存在
    for req_name in REQUIRED_SUBMISSION_FILES:
        req_path = sub_dir / req_name
        if not req_path.exists():
            raise CoverageError(f"提交缺少必需文件或目录: {req_name}")
        if req_name == "artifacts" and not req_path.is_dir():
            raise CoverageError("artifacts 必须为目录")

    # 2. 解析 submission.yaml
    try:
        raw_manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CoverageError(f"读取 submission.yaml 失败: {exc}")

    if not isinstance(raw_manifest, dict):
        raise CoverageError("submission.yaml 根结构必须为字典")

    manifest = SubmissionManifest(**raw_manifest)

    # 校验参赛者编号与目录名称匹配
    if manifest.participant_id != sub_dir.name:
        raise CoverageError(f"提交目录名与参赛者编号不匹配: {sub_dir.name} != {manifest.participant_id}")

    # 校验协议与任务版本匹配
    if manifest.protocol_version != release.protocol.protocol_version:
        raise CoverageError(
            f"协议版本不匹配: {manifest.protocol_version} != {release.protocol.protocol_version}"
        )
    if manifest.task_version != release.protocol.task_version:
        raise CoverageError(
            f"任务版本不匹配: {manifest.task_version} != {release.protocol.task_version}"
        )

    # 校验 training_config_paths
    for train_cfg in manifest.training_config_paths:
        if not isinstance(train_cfg, str):
            raise CoverageError("training_config_paths 必须为字符串路径")
        if train_cfg.startswith("/") or train_cfg.startswith("\\") or ":" in train_cfg:
            raise CoverageError(f"training_config_paths 禁止使用绝对路径或盘符: {train_cfg}")
        norm_cfg = train_cfg.replace("\\", "/")
        if ".." in norm_cfg.split("/"):
            raise CoverageError(f"training_config_paths 禁止相对越界: {train_cfg}")
        cfg_path = sub_dir / norm_cfg
        if not cfg_path.is_file():
            raise CoverageError(f"登记的训练配置文件不存在: {train_cfg}")
        if cfg_path.is_symlink():
            raise CoverageError(f"training_config_paths 禁止使用符号链接: {train_cfg}")
        if hasattr(os.path, "isjunction") and os.path.isjunction(cfg_path):
            raise CoverageError(f"training_config_paths 禁止使用目录联接: {train_cfg}")

    # 3. 校验 entry.py 导出 build_policy
    _check_entrypoint_ast(sub_dir / "entry.py")

    # 4. 校验受保护依赖
    lock_file = sub_dir / "requirements-infer.lock"
    _check_protected_dependencies(lock_file, release.protected_dependencies)

    # 5. 符号链接与外部硬链接安全校验
    artifact_dir = sub_dir / "artifacts"
    for root_p, _, files in os.walk(artifact_dir):
        for f in files:
            fp = Path(root_p) / f
            if fp.is_symlink():
                raise CoverageError(f"产物目录中禁止符号链接: {fp}")

    ino_counts: dict[tuple[int, int], int] = {}
    for root_p, _, files in os.walk(sub_dir):
        for f in files:
            fp = Path(root_p) / f
            if not fp.is_symlink():
                st = fp.stat()
                key = (st.st_dev, st.st_ino)
                ino_counts[key] = ino_counts.get(key, 0) + 1

    for root_p, _, files in os.walk(sub_dir):
        for f in files:
            fp = Path(root_p) / f
            if not fp.is_symlink():
                st = fp.stat()
                key = (st.st_dev, st.st_ino)
                if st.st_nlink > ino_counts[key]:
                    raise CoverageError(f"检测到外部硬链接引用: {fp}")

    # 6. 先创建独立隔离快照，消除 TOCTOU 竞态
    snapshot_id = f"snap-{uuid.uuid4().hex[:16]}"
    work_base = Path(os.environ.get("COVERAGE_WORK_DIR", Path(__file__).resolve().parents[1] / ".work"))
    work_snapshot_dir = work_base / "snapshots" / snapshot_id
    work_snapshot_dir.mkdir(parents=True, exist_ok=True)

    # 复制所有文件（解除内部硬链接别名）
    total_sub_bytes = 0
    for root_p, dirs, files in os.walk(sub_dir):
        rel_dir = Path(root_p).relative_to(sub_dir)
        target_dir = work_snapshot_dir / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            src_file = Path(root_p) / f
            dst_file = target_dir / f
            shutil.copyfile(src_file, dst_file)
            total_sub_bytes += dst_file.stat().st_size

    if total_sub_bytes > limits.submission_bytes_total:
        shutil.rmtree(work_snapshot_dir, ignore_errors=True)
        raise CoverageError(
            f"提交总字节数超过限制: {total_sub_bytes} > {limits.submission_bytes_total}"
        )

    # 7. 在隔离快照上校验受保护依赖
    lock_file = work_snapshot_dir / "requirements-infer.lock"
    _check_protected_dependencies(lock_file, release.protected_dependencies)

    # 8. 在隔离快照上校验产物清单与文件一致性
    snap_artifact_dir = work_snapshot_dir / "artifacts"
    seen_case_folded: set[str] = set()
    manifest_paths_set: set[str] = set()
    artifact_hashes: dict[str, str] = {}
    total_artifact_bytes = 0

    for entry in manifest.checkpoint_manifest:
        norm_path = entry.path.replace("\\", "/")
        folded = norm_path.lower()
        if folded in seen_case_folded:
            raise CoverageError(f"产物清单包含重复路径（含大小写折叠）: {entry.path}")
        seen_case_folded.add(folded)
        manifest_paths_set.add(norm_path)

        actual_file = work_snapshot_dir / norm_path
        if not actual_file.is_file():
            raise CoverageError(f"登记的产物文件不存在: {entry.path}")
        
        actual_size = actual_file.stat().st_size
        if entry.size_bytes != actual_size:
            raise CoverageError(f"产物文件大小不匹配: {entry.path}, 预期 {entry.size_bytes}, 实际 {actual_size}")
            
        actual_hash = _sha256_file(actual_file)
        if entry.sha256 != actual_hash:
            raise CoverageError(f"产物文件哈希不匹配: {entry.path}, 预期 {entry.sha256}, 实际 {actual_hash}")

        artifact_hashes[norm_path] = actual_hash
        total_artifact_bytes += actual_size

    # 校验 artifacts 目录下的所有文件都必须被登记
    for root_p, _, files in os.walk(snap_artifact_dir):
        for f in files:
            fp = Path(root_p) / f
            rel = fp.relative_to(work_snapshot_dir).as_posix()
            if rel not in manifest_paths_set:
                raise CoverageError(f"发现未在清单中登记的产物文件: {rel}")

    # 校验产物体积预算限制
    if total_artifact_bytes > limits.artifact_bytes_total:
        shutil.rmtree(work_snapshot_dir, ignore_errors=True)
        raise CoverageError(
            f"产物总字节数超过限制: {total_artifact_bytes} > {limits.artifact_bytes_total}"
        )

    # 9. 计算快照代码树哈希与锁文件哈希
    tree_records: list[tuple[str, str]] = []
    for root_p, _, files in os.walk(work_snapshot_dir):
        for f in sorted(files):
            fp = Path(root_p) / f
            rel_p = fp.relative_to(work_snapshot_dir).as_posix()
            tree_records.append((rel_p, _sha256_file(fp)))

    tree_records.sort(key=lambda x: x[0])
    code_tree_payload = yaml.safe_dump(tree_records, sort_keys=True).encode("utf-8")
    code_tree_hash = hashlib.sha256(code_tree_payload).hexdigest()

    inference_lock_hash = _sha256_file(work_snapshot_dir / "requirements-infer.lock")

    # 清单摘要取自隔离快照，与代码、模型、锁保持同一来源
    manifest_rel = manifest_file.relative_to(sub_dir)
    snapshot_manifest = work_snapshot_dir / manifest_rel
    if not snapshot_manifest.is_file():
        shutil.rmtree(work_snapshot_dir, ignore_errors=True)
        raise CoverageError(f"隔离快照中缺少提交清单: {manifest_rel}")
    manifest_sha256 = _sha256_file(snapshot_manifest)

    return ValidatedSubmission(
        manifest=manifest,
        root=work_snapshot_dir,
        manifest_path=manifest_file,
        artifact_hashes=artifact_hashes,
        inference_lock_hash=inference_lock_hash,
        code_tree_hash=code_tree_hash,
        manifest_sha256=manifest_sha256,
        source_commit=source_commit,
        working_tree_dirty=working_tree_dirty,
        snapshot_id=snapshot_id,
        protocol_version=release.protocol.protocol_version,
        task_version=release.protocol.task_version,
    )
