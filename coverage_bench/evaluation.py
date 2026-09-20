"""评测主流程：跑完套件全部回合、聚合评分并原子发布 result.json 等产物；含官方核验入口。"""
import hashlib
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from coverage_bench.archive_types import FreezeReceipt
from coverage_bench.deadline import phase_deadline
from coverage_bench.errors import CoverageError
from coverage_bench.hardware import _default_hardware_profile
from coverage_bench.protocol import ResourceLimits, get_protocol_spec
from coverage_bench.results import (
    EpisodeRecord,
    ErrorRecord,
    EvaluationResult,
    HardwareProfile,
    write_episode_records,
    write_result,
)
from coverage_bench.runtime import DeadlineFactory, run_episode
from coverage_bench.schedules import SeedSchedule
from coverage_bench.scoring import (
    ScoringConfig,
    aggregate_episodes,
    calculate_performance_score,
)
from coverage_bench.release import ReleaseBundle, compute_release_bundle_hash
from coverage_bench.submission import ValidatedSubmission
from coverage_bench.suites import EvaluationSuite


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# _default_hardware_profile 已移至 coverage_bench/hardware.py（Task 14 宿主侧
# 与环境栈解耦）；此处保留同名引入使既有调用点（evaluate_submission 及容器内
# scripts/evaluate_one.py 链路）零改动


def evaluate_submission(
    submission: ValidatedSubmission,
    suite: EvaluationSuite,
    schedule: SeedSchedule,
    scoring: ScoringConfig,
    limits: ResourceLimits,
    output_dir: Path,
    provenance: str = "self_reported",
    release: Optional[ReleaseBundle] = None,
    *,
    deadline_factory: DeadlineFactory = phase_deadline,
    hardware_profile: Optional[HardwareProfile] = None,
) -> EvaluationResult:
    """在空输出目录中执行整套评测：逐回合运行、超时熔断、聚合评分并发布结果。"""
    out = Path(output_dir).resolve()
    if out.exists():
        existing_items = list(out.iterdir())
        if len(existing_items) > 0:
            raise CoverageError(
                f"输出目录必须为空目录或不存在，禁止写入非空目录: {out}",
                code="OUTPUT_DIRECTORY_NOT_EMPTY",
            )
    else:
        out.mkdir(parents=True, exist_ok=True)

    # 暂存目录建在输出目录内部子目录：容器以 --read-only 挂根运行时 out.parent
    # 即容器根 /，不可写，暂存建在 out.parent 会在评测开始前触发 EROFS；
    # out 与 out/.staging_tmp 同属 /output 挂载（宿主直跑时同盘），同一文件系统，
    # 发布时可对单文件用 os.replace 保持原子性。
    # 注意：空目录检查在上方已完成（先于暂存创建），上一次运行被杀残留的
    # .staging_tmp 会触发 OUTPUT_DIRECTORY_NOT_EMPTY 快速失败，属预期语义
    staging_out = out / f".staging_tmp_{uuid.uuid4().hex[:8]}"
    staging_out.mkdir(parents=True, exist_ok=True)

    try:
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        started_at = datetime.now(timezone.utc).isoformat()
        wall_start = time.perf_counter()

        eval_deadline = (
            wall_start + (limits.evaluation_ms / 1000.0)
            if limits.evaluation_ms > 0
            else float("inf")
        )

        all_records: list[EpisodeRecord] = []
        all_errors: list[ErrorRecord] = []
        all_timings: list[float] = []
        overall_status = "ok"

        # 依次执行套件中所有组别、用例与重复次数
        for group in suite.groups:
            for case in group.cases:
                for rep in range(group.policy_repeats):
                    if time.perf_counter() > eval_deadline:
                        overall_status = "timeout"
                        all_errors.append(
                            ErrorRecord(
                                code="TIMEOUT",
                                stage="evaluation",
                                retryable=False,
                                owner="participant",
                                case_id=case.case_id,
                                repeat_index=rep,
                                message="评测执行总墙钟时间超出 evaluation_ms 预算限额",
                            )
                        )
                        break

                    seed_records = schedule.get_records(case.case_id, rep)
                    episode_run = run_episode(
                        submission=submission,
                        case=case,
                        seed_records=seed_records,
                        limits=limits,
                        run_id=run_id,
                        attempt_index=0,
                        deadline_factory=deadline_factory,
                    )

                    all_timings.extend(episode_run.act_timings_ms)
                    if episode_run.errors:
                        all_errors.extend(episode_run.errors)

                    if episode_run.record.status != "ok":
                        overall_status = episode_run.record.status
                        all_records.append(episode_run.record)
                    else:
                        accepted_rec = episode_run.record.model_copy(update={"accepted": True})
                        all_records.append(accepted_rec)

                if overall_status != "ok" and time.perf_counter() > eval_deadline:
                    break
            if overall_status != "ok" and time.perf_counter() > eval_deadline:
                break

        # 计算聚合评分
        if overall_status == "ok":
            group_metrics = aggregate_episodes(
                tuple(all_records), suite, scoring, timings=np.array(all_timings, dtype=np.float32)
            )
            performance_score = calculate_performance_score(group_metrics, scoring)
            # 计算总体 Bootstrap 置信区间
            if all(gm.ci95_j is not None for gm in group_metrics.values()):
                score_low = sum(
                    1000.0 * g.weight * (group_metrics[g.group_id].ci95_j[0] - g.random_anchor)
                    / (g.learning_anchor - g.random_anchor)
                    for g in scoring.groups
                )
                score_high = sum(
                    1000.0 * g.weight * (group_metrics[g.group_id].ci95_j[1] - g.random_anchor)
                    / (g.learning_anchor - g.random_anchor)
                    for g in scoring.groups
                )
                performance_score_ci95 = (float(score_low), float(score_high))
            else:
                performance_score_ci95 = None
        else:
            group_metrics = {}
            performance_score = None
            performance_score_ci95 = None

        # 写入 episodes.csv 与 timings.npz 到暂存目录，并计算文件摘要
        episodes_file = staging_out / "episodes.csv"
        write_episode_records(tuple(all_records), episodes_file)
        episodes_hash = _file_sha256(episodes_file)

        timings_file = staging_out / "timings.npz"
        np.savez_compressed(timings_file, timings=np.array(all_timings, dtype=np.float32))
        timings_hash = _file_sha256(timings_file)

        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        finished_at = datetime.now(timezone.utc).isoformat()
        spec = get_protocol_spec()

        if release is not None:
            off_commit = release.official_commit
            rel_state = release.release_state
            rel_bundle_hash = compute_release_bundle_hash(release)
        else:
            off_commit = None
            rel_bundle_hash = "development-release"
            rel_state = "development"

        result = EvaluationResult(
            result_schema_version="coverage-result/1.0",
            run_id=run_id,
            participant_id=submission.manifest.participant_id,
            provenance=provenance,
            status=overall_status,
            source_commit=submission.source_commit,
            working_tree_dirty=submission.working_tree_dirty,
            code_tree_hash=submission.code_tree_hash,
            inference_lock_hash=submission.inference_lock_hash,
            artifact_hashes=submission.artifact_hashes,
            official_commit=off_commit,
            release_bundle_hash=rel_bundle_hash,
            release_state=rel_state,
            suite_visibility=suite.visibility,
            protocol_version=spec.protocol_version,
            task_version=spec.task_version,
            score_version=scoring.score_version,
            suite_id=suite.suite_id,
            suite_hash=suite.suite_hash,
            seed_schedule_id=schedule.schedule_id,
            hardware_profile=hardware_profile if hardware_profile is not None else _default_hardware_profile(),
            runtime_limits=limits,
            group_metrics=group_metrics,
            performance_score=performance_score,
            performance_score_ci95=performance_score_ci95,
            episode_records_path="episodes.csv",
            episode_records_hash=episodes_hash,
            timings_path="timings.npz",
            timings_hash=timings_hash,
            errors=all_errors,
            started_at=started_at,
            finished_at=finished_at,
        )

        write_result(result, staging_out)

        # 逐文件提升到 out 根目录后删除暂存子目录：结果完整写完才出现在最终位置；
        # out 与暂存同挂载点（容器内同属 /output，宿主直跑同盘），单文件 os.replace
        # 保持原子发布，跨文件系统等退化场景回退为复制+删除。
        # 二阶场景：提升中途进程被杀会在 out/ 留下部分文件与 .staging_tmp 残留，
        # 宿主侧按设计 §8 结果损坏/缺失分类兜底（RESULT_CORRUPT / EVALUATION_TIMEOUT），
        # 此处不引入重试逻辑；残留的 .staging_tmp 会被下一次评测的空目录检查拒绝
        for item in sorted(staging_out.iterdir()):
            target = out / item.name
            try:
                os.replace(item, target)
            except OSError:
                shutil.move(str(item), str(target))
        staging_out.rmdir()

        return result
    except Exception:
        if staging_out.exists():
            shutil.rmtree(staging_out, ignore_errors=True)
        raise


def verify_submission(
    submission: ValidatedSubmission,
    receipt: FreezeReceipt,
    suite: EvaluationSuite,
    schedule: SeedSchedule,
    scoring: ScoringConfig,
    limits: ResourceLimits,
    output_dir: Path,
    release: Optional[ReleaseBundle] = None,
    *,
    deadline_factory: DeadlineFactory = phase_deadline,
    hardware_profile: Optional[HardwareProfile] = None,
) -> EvaluationResult:
    """官方核验：先比对冻结接收记录与提交快照的全部哈希，再用 private 套件复测。"""
    if receipt.freeze_record.participant_id != submission.manifest.participant_id:
        raise CoverageError(
            f"冻结接收记录的选手编号不匹配: {receipt.freeze_record.participant_id} != {submission.manifest.participant_id}",
            code="RECEIPT_PARTICIPANT_MISMATCH",
        )

    if receipt.freeze_record.code_tree_hash != submission.code_tree_hash:
        raise CoverageError(
            f"冻结记录源码树哈希不匹配: {receipt.freeze_record.code_tree_hash} != {submission.code_tree_hash}",
            code="RECEIPT_CODE_TREE_MISMATCH",
        )

    if receipt.freeze_record.inference_lock_hash != submission.inference_lock_hash:
        raise CoverageError(
            f"冻结记录推理依赖锁哈希不匹配: {receipt.freeze_record.inference_lock_hash} != {submission.inference_lock_hash}",
            code="RECEIPT_LOCK_MISMATCH",
        )

    if receipt.freeze_record.artifact_hashes != submission.artifact_hashes:
        raise CoverageError(
            "冻结记录模型产物哈希不匹配",
            code="RECEIPT_ARTIFACT_MISMATCH",
        )

    if receipt.freeze_record.submission_manifest_hash != submission.manifest_sha256:
        raise CoverageError(
            "冻结记录提交清单哈希与实际快照不一致: "
            f"{receipt.freeze_record.submission_manifest_hash} != {submission.manifest_sha256}",
            code="RECEIPT_MANIFEST_MISMATCH",
        )

    if receipt.freeze_record.working_tree_dirty:
        raise CoverageError(
            "冻结记录显示 Git 工作区存在未提交修改，禁止核验",
            code="RECEIPT_WORKING_TREE_DIRTY",
        )

    if submission.working_tree_dirty:
        raise CoverageError(
            "当前提交工作区存在未提交修改，不能按冻结快照核验",
            code="RECEIPT_WORKING_TREE_DIRTY",
        )

    if not receipt.freeze_record.source_commit:
        raise CoverageError(
            "冻结记录缺少有效 source_commit，无法确立提交身份",
            code="RECEIPT_COMMIT_MISSING",
        )

    if not submission.source_commit or receipt.freeze_record.source_commit != submission.source_commit:
        raise CoverageError(
            "冻结接收记录的提交身份与实际快照不一致: "
            f"{receipt.freeze_record.source_commit} != {submission.source_commit}",
            code="RECEIPT_COMMIT_MISMATCH",
        )

    if release is None or release.release_state != "frozen":
        raise CoverageError(
            "官方核验必须提供 release_state 为 frozen 的发布包",
            code="VERIFICATION_RELEASE_INVALID",
        )

    if suite.visibility != "private":
        raise CoverageError(
            "官方核验必须使用 private 套件进行评测",
            code="VERIFICATION_SUITE_INVALID",
        )

    return evaluate_submission(
        submission=submission,
        suite=suite,
        schedule=schedule,
        scoring=scoring,
        limits=limits,
        output_dir=output_dir,
        provenance="official_verified",
        release=release,
        deadline_factory=deadline_factory,
        hardware_profile=hardware_profile,
    )
