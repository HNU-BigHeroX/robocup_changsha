"""提交冻结：把校验通过的提交固化为 FreezeRecord（Git 提交、产物与依赖锁哈希）。

冻结时会重新核验实时仓库，任何校验之后发生的改动都会被拒绝。
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from coverage_bench.archive_types import FreezeRecord
from coverage_bench.errors import CoverageError
from coverage_bench.submission import ValidatedSubmission, working_tree_dirty_status


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def freeze_submission(
    submission: ValidatedSubmission,
    tag: str,
    output_path: Optional[Path] = None,
) -> FreezeRecord:
    """核验 Git 标签与工作区后生成 FreezeRecord，output_path 非空时同时落盘 JSON。"""
    root = submission.root.resolve()
    git_dir = submission.manifest_path.parent.resolve()

    # 1. 验证 Git 工作区与标签
    if submission.working_tree_dirty:
        raise CoverageError(
            "Git 工作区存在未提交的修改，无法进行提交冻结",
            code="GIT_WORKING_TREE_DIRTY",
        )
    if not submission.source_commit:
        raise CoverageError(
            "无法确立 Git HEAD 提交身份",
            code="GIT_COMMIT_MISSING",
        )

    source_commit = submission.source_commit
    code_tree_hash = submission.code_tree_hash

    # 冻结时重新核验实时仓库，拒绝校验之后发生的任何改动
    live_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(git_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if live_commit.returncode != 0 or live_commit.stdout.strip() != source_commit:
        raise CoverageError(
            "校验之后 Git HEAD 已改变，冻结内容与校验提交不再一致",
            code="GIT_COMMIT_CHANGED",
        )

    if working_tree_dirty_status(git_dir):
        raise CoverageError(
            "校验之后 Git 工作区出现未提交或被忽略的内容，无法进行提交冻结",
            code="GIT_WORKING_TREE_DIRTY",
        )

    tag_verify = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"],
        cwd=str(git_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if tag_verify.returncode != 0:
        raise CoverageError(f"指定的 Git 标签不存在: {tag}", code="GIT_TAG_NOT_FOUND")
    tag_commit = tag_verify.stdout.strip()
    if tag_commit != source_commit:
        raise CoverageError("指定的 Git 标签未指向当前完整提交", code="GIT_TAG_MISMATCH")

    remote_res = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        cwd=str(git_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    repo_url = remote_res.stdout.strip() if remote_res.returncode == 0 and remote_res.stdout.strip() else f"local://{git_dir.name}"

    # 2. 收集模型产物哈希
    artifacts_dir = root / "artifacts"
    artifact_hashes = {}
    if artifacts_dir.is_dir():
        for f in sorted(artifacts_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(root).as_posix()
                artifact_hashes[rel] = _file_sha256(f)

    if artifact_hashes != submission.artifact_hashes:
        raise CoverageError("冻结提取的产物哈希与校验提交不一致", code="ARTIFACT_HASH_MISMATCH")

    # 3. 计算依赖锁与清单哈希，全部取自同一份隔离快照
    lock_file = root / submission.manifest.inference_lock
    if not lock_file.is_file():
        raise CoverageError(f"推理依赖锁文件不存在: {lock_file}", code="LOCK_FILE_MISSING")
    inference_lock_hash = _file_sha256(lock_file)

    manifest_hash = submission.manifest_sha256

    created_at = datetime.now(timezone.utc).isoformat()

    record = FreezeRecord(
        freeze_schema_version="coverage-freeze/1.0",
        participant_id=submission.manifest.participant_id,
        repository_url=repo_url,
        final_tag=tag,
        source_commit=source_commit,
        code_tree_hash=code_tree_hash,
        artifact_hashes=artifact_hashes,
        inference_lock_hash=inference_lock_hash,
        submission_manifest_hash=manifest_hash,
        protocol_version=submission.protocol_version,
        task_version=submission.task_version,
        created_at=created_at,
        working_tree_dirty=submission.working_tree_dirty,
    )

    if output_path is not None:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(record.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

    return record
