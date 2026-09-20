import argparse
import json
import sys
from pathlib import Path

# 引导仓库根目录进入模块搜索路径，保证 `python scripts/check_submission.py` 可直接运行
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from coverage_bench.audit import AuditFinding, audit_submission
from coverage_bench.errors import CoverageError
from coverage_bench.submission import validate_submission


def _finding_to_dict(finding: AuditFinding) -> dict:
    return {
        "code": finding.code,
        "path": finding.path,
        "line": finding.line,
        "symbol": finding.symbol,
        "detail": finding.detail,
    }


def _summarize_submission(submission_root: Path) -> dict:
    # 提交目录摘要：路径、文件数量与总字节数等基础事实
    file_count = 0
    total_bytes = 0
    for file_path in submission_root.rglob("*"):
        if file_path.is_file():
            file_count += 1
            total_bytes += file_path.stat().st_size
    return {
        "path": str(submission_root),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def check_submission_cli(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coverage Bench 提交预检入口: 结构校验 + 白名单与静态审核")
    parser.add_argument("--submission", type=str, required=True, help="参赛提交目录 (含 submission.yaml)")
    parser.add_argument("--output", type=str, required=True, help="审核报告输出目录")
    args = parser.parse_args(argv)

    submission_root = Path(args.submission).resolve()
    manifest_path = submission_root / "submission.yaml"
    if not manifest_path.is_file():
        sys.stderr.write(f"错误: 提交清单文件不存在: {manifest_path}\n")
        return 2

    # 1. validate_submission 做结构与摘要校验（必需文件、清单一致性、哈希、软硬链接、体积预算）
    try:
        validated = validate_submission(manifest_path)
    except CoverageError as exc:
        sys.stderr.write(f"提交校验失败 [{exc.code}]: {exc.message}\n")
        return 2

    # 2. audit_submission 做白名单与静态审核（后缀三类处置 + ast 分析，基于原始提交目录）
    try:
        report = audit_submission(submission_root)
    except CoverageError as exc:
        sys.stderr.write(f"静态审核失败 [{exc.code}]: {exc.message}\n")
        return 2

    # 3. 构造并写出 audit-report.json：无论是否存在硬拒绝都持久化报告，
    #    满足 AT-AUD-04「提交无效并保留证据」的要求
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "submission": _summarize_submission(submission_root),
        "rejections": [_finding_to_dict(finding) for finding in report.rejections],
        "scan_hits": [_finding_to_dict(finding) for finding in report.scan_hits],
        "review": report.review,
        "digest": {
            "participant_id": validated.manifest.participant_id,
            "snapshot_id": validated.snapshot_id,
            "manifest_sha256": validated.manifest_sha256,
            "code_tree_hash": validated.code_tree_hash,
            "inference_lock_hash": validated.inference_lock_hash,
            "protocol_version": validated.protocol_version,
            "task_version": validated.task_version,
            "source_commit": validated.source_commit,
            "working_tree_dirty": validated.working_tree_dirty,
        },
    }
    report_path = output_dir / "audit-report.json"
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 4. 存在硬拒绝：逐条打印后退出码 2（报告已写出作为持久证据）
    if report.rejections:
        for finding in report.rejections:
            sys.stderr.write(f"硬拒绝 [{finding.code}] {finding.path}: {finding.detail}\n")
        return 2

    print(f"审核通过: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(check_submission_cli())
