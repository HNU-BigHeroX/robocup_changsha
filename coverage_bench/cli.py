"""官方命令行入口：提交预检、冻结、发布包校验、单人评测与批量评分。"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from coverage_bench.archive import freeze_submission
from coverage_bench.config import PublicConfig, ScenarioConfig, TaskConfig
from coverage_bench.errors import CoverageError
from coverage_bench.evaluation import evaluate_submission
from coverage_bench.protocol import ResourceLimits
from coverage_bench.runtime import run_episode
from coverage_bench.schedules import PolicySeedRecord, create_seed_schedule
from coverage_bench.scoring import ScoringConfig, default_scoring_config
from coverage_bench.submission import validate_submission
from coverage_bench.suites import ScenarioCase, load_suite


def _default_scoring_config(suite: Any) -> ScoringConfig:
    # 兼容别名：实现已提升为 scoring.default_scoring_config 公共 helper，
    # 容器内评测与宿主侧复算共用同一评分配置推导，保证口径一致
    return default_scoring_config(suite)


def check_submission_cli(argv: Optional[List[str]] = None) -> int:
    """提交预检：静态校验后跑 N=3/N=8 两个短回合，退出码 0=通过、2=参数错、3=预检失败。"""
    parser = argparse.ArgumentParser(description="Coverage Bench 提交预检工具")
    parser.add_argument("--submission", type=str, required=True, help="参赛提交清单路径 (submission.yaml)")
    parser.add_argument("--output", type=str, default=None, help="预检报告输出路径")
    args = parser.parse_args(argv)

    sub_path = Path(args.submission).resolve()
    if not sub_path.is_file():
        sys.stderr.write(f"错误: 提交清单文件不存在: {args.submission}\n")
        return 2

    try:
        submission = validate_submission(sub_path)
    except CoverageError as e:
        sys.stderr.write(f"提交静态校验失败: {e}\n")
        return 2

    # 执行短回合预检 (N=3 与 N=8)，进程内执行
    limits = ResourceLimits()

    precheck_passed = True
    precheck_error: str | None = None

    for agent_n in (3, 8):
        cfg = TaskConfig(
            config_schema_version="coverage-task/1.0",
            protocol_version="coverage-policy/1.0",
            task_version="coverage-task/1.0",
            release_state="development",
            public=PublicConfig(
                map_half_extent=1.0,
                dt=0.1,
                robot_radius=0.05,
                robot_mass=1.0,
                drive_force=1.0,
                damping=0.25,
                robot_max_speed=1.0,
                contact_force=50.0,
                contact_margin=0.01,
                target_radius=0.15,
                target_max_speed=0.5,
                sense_radius=0.75,
                motion_kind="piecewise_heading_reflect",
                turn_interval_steps=(10, 20),
                target_speed_fraction=(0.2, 1.0),
                robot_boundary="clamp_outward_velocity",
                target_boundary="reflect_cover_inset",
            ),
            scenario=ScenarioConfig(
                layout_kind="uniform",
                robot_min_gap=0.02,
                sampling_attempt_limit=1000,
                crossing_band_fraction=0.3,
                crossing_angle_jitter=0.2,
                cluster_radius=0.3,
            ),
            num_agents=agent_n,
            num_targets=2,
            horizon=3,
            collision_weight=0.1,
        )
        case = ScenarioCase(
            case_id=f"precheck-{agent_n}",
            group_id="precheck",
            task_config=cfg,
            scenario_seed=42,
        )
        seeds = tuple(
            PolicySeedRecord(
                case_id=case.case_id,
                repeat_index=0,
                agent_index=i,
                build_seed=1000 + i,
                policy_seed=2000 + i,
            )
            for i in range(agent_n)
        )

        try:
            run_res = run_episode(
                submission=submission,
                case=case,
                seed_records=seeds,
                limits=limits,
                run_id=f"precheck-{agent_n}",
                attempt_index=0,
            )
            if run_res.record.status != "ok":
                precheck_passed = False
                precheck_error = f"N={agent_n} 短回合执行未成功: status={run_res.record.status}"
                break
        except Exception as exc:
            precheck_passed = False
            precheck_error = f"N={agent_n} 运行时异常: {exc}"
            break

    report = {
        "check_schema_version": "coverage-check/1.0",
        "participant_id": submission.manifest.participant_id,
        "status": "ok" if precheck_passed else "failed",
        "checks": {
            "static_structure": True,
            "manifest_integrity": True,
            "short_episode_n3": precheck_passed,
            "short_episode_n8": precheck_passed,
        },
        "error": precheck_error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if args.output:
        out_p = Path(args.output).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0 if precheck_passed else 3


def freeze_submission_cli(argv: Optional[List[str]] = None) -> int:
    """提交冻结入口：校验后按指定 Git 标签生成冻结记录。"""
    parser = argparse.ArgumentParser(description="Coverage Bench 提交冻结工具")
    parser.add_argument("--submission", type=str, required=True, help="参赛提交清单路径 (submission.yaml)")
    parser.add_argument("--tag", type=str, default="v1.0-final", help="冻结 Git 标签")
    parser.add_argument("--output", type=str, default=None, help="冻结凭据输出路径")
    args = parser.parse_args(argv)

    sub_path = Path(args.submission).resolve()
    if not sub_path.is_file():
        sys.stderr.write(f"错误: 提交清单文件不存在: {args.submission}\n")
        return 2

    try:
        submission = validate_submission(sub_path)
    except CoverageError as e:
        sys.stderr.write(f"提交校验失败: {e}\n")
        return 2

    out_p = Path(args.output).resolve() if args.output else None
    try:
        freeze_submission(submission, tag=args.tag, output_path=out_p)
    except CoverageError as e:
        sys.stderr.write(f"冻结执行失败: {e}\n")
        return 2

    return 0


def check_release_cli(argv: Optional[List[str]] = None) -> int:
    """发布包校验入口：逐项输出校验结果并可选落盘记录。"""
    parser = argparse.ArgumentParser(description="Coverage Bench 发布包校验工具")
    parser.add_argument("--bundle", type=str, required=True, help="发布包清单路径 (release.json)")
    parser.add_argument("--output", type=str, default=None, help="校验记录输出路径")
    args = parser.parse_args(argv)

    from coverage_bench.release import validate_release

    bundle_path = Path(args.bundle).resolve()
    report = validate_release(bundle_path)

    failed = [c for c in report.checks if not c.passed]
    for check in report.checks:
        state = "通过" if check.passed else "失败"
        sys.stderr.write(f"[{state}] {check.name}: {check.message}\n")

    record = {
        "check_schema_version": "coverage-release-check/1.0",
        "bundle_path": str(bundle_path),
        "valid": report.valid,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": [
            {"name": c.name, "passed": c.passed, "message": c.message}
            for c in report.checks
        ],
    }
    if args.output:
        out_p = Path(args.output).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    if not report.valid:
        sys.stderr.write(f"发布包校验失败，共 {len(failed)} 项未通过\n")
        return 2
    return 0


def evaluate_cli(argv: Optional[List[str]] = None) -> int:
    """单人评测入口：可选接入发布包（评分/种子安排/资源限额以发布组件为准）。"""
    parser = argparse.ArgumentParser(description="Coverage Bench 评测执行入口")
    parser.add_argument("--submission", type=str, required=True, help="参赛提交清单文件路径 (submission.yaml)")
    parser.add_argument("--suite", type=str, required=True, help="评测套件配置文件路径")
    parser.add_argument("--output", type=str, required=True, help="评测结果输出目录")
    parser.add_argument("--release", type=str, default=None, help="发布包清单路径")
    args = parser.parse_args(argv)

    sub_path = Path(args.submission).resolve()
    if not sub_path.is_file():
        sys.stderr.write(f"错误: 提交文件不存在: {args.submission}\n")
        return 2

    suite_path = Path(args.suite).resolve()
    if not suite_path.is_file():
        sys.stderr.write(f"错误: 套件文件不存在: {args.suite}\n")
        return 2

    output_dir = Path(args.output).resolve()

    release_bundle = None
    if args.release:
        rel_p = Path(args.release).resolve()
        if not rel_p.is_file():
            sys.stderr.write(f"错误: 发布包文件不存在: {args.release}\n")
            return 2
        from coverage_bench.release import load_release
        try:
            release_bundle = load_release(rel_p)
        except CoverageError as exc:
            sys.stderr.write(f"发布包校验失败: {exc}\n")
            return 2

    try:
        submission = validate_submission(sub_path, release=release_bundle)
    except CoverageError as e:
        sys.stderr.write(f"提交校验错误: {e}\n")
        return 2

    try:
        suite = load_suite(suite_path, release=release_bundle)
    except CoverageError as e:
        sys.stderr.write(f"套件配置错误: {e}\n")
        return 2

    if release_bundle is not None:
        from coverage_bench.release import (
            load_release_runtime_limits,
            load_release_schedule,
            load_release_scoring,
        )
        try:
            scoring = load_release_scoring(release_bundle)
            schedule = load_release_schedule(release_bundle)
            limits = load_release_runtime_limits(release_bundle)
        except CoverageError as exc:
            sys.stderr.write(f"发布组件加载失败: {exc}\n")
            return 2
        if suite.score_version != scoring.score_version:
            sys.stderr.write(
                f"套件与发布包评分配置版本不一致: {suite.score_version} != {scoring.score_version}\n"
            )
            return 2
        if suite.task_version != release_bundle.protocol.task_version:
            sys.stderr.write(
                f"套件与发布包任务版本不一致: {suite.task_version} != {release_bundle.protocol.task_version}\n"
            )
            return 2
        missing = [
            (case.case_id, rep)
            for group in suite.groups for case in group.cases
            for rep in range(group.policy_repeats)
            if not schedule.get_records(case.case_id, rep)
        ]
        if missing:
            sys.stderr.write(f"发布包公开安排未覆盖套件用例: {missing[:5]}\n")
            return 2
    else:
        schedule = create_seed_schedule(suite, policy_entropy=20260918)
        scoring = _default_scoring_config(suite)
        limits = ResourceLimits()

    try:
        result = evaluate_submission(
            submission=submission,
            suite=suite,
            schedule=schedule,
            scoring=scoring,
            limits=limits,
            output_dir=output_dir,
            release=release_bundle,
        )
    except CoverageError as e:
        sys.stderr.write(f"评测执行致命错误: {e}\n")
        return 4

    if result.status == "ok":
        return 0
    # resource_limit_exceeded 为已删除的 worker 进程模型遗留状态，核实无生产者后移除，
    # 与 scripts/evaluate_one.py 的退出码映射保持一致
    if result.status in ("load_error", "timeout", "protocol_error", "runtime_error"):
        return 3
    return 4


def batch_score_cli(argv: Optional[List[str]] = None) -> int:
    """批量评分入口：遍历提交根目录逐人评测，哈希一致时复用缓存结果，并导出排行榜。"""
    parser = argparse.ArgumentParser(description="Coverage Bench 批量评分入口")
    parser.add_argument("--submissions", type=str, required=True, help="参赛提交根目录")
    parser.add_argument("--suite", type=str, required=True, help="评测套件配置文件路径")
    parser.add_argument("--output", type=str, required=True, help="评分结果输出目录")
    args = parser.parse_args(argv)

    sub_dir = Path(args.submissions).resolve()
    if not sub_dir.is_dir():
        sys.stderr.write(f"错误: 提交目录不存在: {args.submissions}\n")
        return 2

    suite_path = Path(args.suite).resolve()
    if not suite_path.is_file():
        sys.stderr.write(f"错误: 套件文件不存在: {args.suite}\n")
        return 2

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        suite = load_suite(suite_path)
    except CoverageError as e:
        sys.stderr.write(f"套件配置错误: {e}\n")
        return 2

    schedule = create_seed_schedule(suite, policy_entropy=20260918)
    scoring = _default_scoring_config(suite)
    limits = ResourceLimits()

    candidate_manifests = sorted(
        p for p in sub_dir.rglob("submission.yaml")
        if p.parent.name != "_template" and not p.parent.name.startswith(".")
    )
    if not candidate_manifests:
        sys.stderr.write(f"错误: 目录中未找到任何 submission.yaml: {args.submissions}\n")
        return 2

    results = []
    any_failure = False
    for manifest_path in candidate_manifests:
        p_id = manifest_path.parent.name
        p_out = output_dir / p_id
        try:
            submission = validate_submission(manifest_path)
            res = None
            if (p_out / "result.json").is_file():
                from coverage_bench.results import read_result
                cached_res = read_result(p_out / "result.json")
                # 核对身份与套件哈希、模型哈希，确保能够安全复用
                if (
                    cached_res.participant_id == submission.manifest.participant_id
                    and cached_res.suite_id == suite.suite_id
                    and cached_res.suite_hash == suite.suite_hash
                    and cached_res.seed_schedule_id == schedule.schedule_id
                    and cached_res.code_tree_hash == submission.code_tree_hash
                    and cached_res.inference_lock_hash == submission.inference_lock_hash
                    and cached_res.artifact_hashes == submission.artifact_hashes
                ):
                    res = cached_res

            if res is None:
                if p_out.exists():
                    shutil.rmtree(p_out, ignore_errors=True)
                res = evaluate_submission(
                    submission=submission,
                    suite=suite,
                    schedule=schedule,
                    scoring=scoring,
                    limits=limits,
                    output_dir=p_out,
                )
            results.append(res)
            if res.status != "ok":
                any_failure = True
        except Exception as e:
            any_failure = True
            sys.stderr.write(f"处理选手 {p_id} 评测时异常: {e}\n")

    # 导出排行榜
    leaderboard_data = {
        "leaderboard_schema_version": "coverage-leaderboard/1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite_id": suite.suite_id,
        "entries": [
            {
                "participant_id": r.participant_id,
                "status": r.status,
                "performance_score": r.performance_score,
                "run_id": r.run_id,
            }
            for r in results
        ],
    }
    (output_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if any_failure:
        return 6
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """子命令分发入口。"""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Coverage Bench 官方命令行管理工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_eval = subparsers.add_parser("evaluate", help="执行单人评测")
    p_eval.add_argument("--submission", type=str, required=True)
    p_eval.add_argument("--suite", type=str, required=True)
    p_eval.add_argument("--output", type=str, required=True)
    p_eval.add_argument("--release", type=str, default=None)

    p_batch = subparsers.add_parser("batch", help="执行批量评分")
    p_batch.add_argument("--submissions", type=str, required=True)
    p_batch.add_argument("--suite", type=str, required=True)
    p_batch.add_argument("--output", type=str, required=True)

    p_check = subparsers.add_parser("check", help="提交静态与运行时预检")
    p_check.add_argument("--submission", type=str, required=True)
    p_check.add_argument("--output", type=str, default=None)

    p_freeze = subparsers.add_parser("freeze", help="提交版本冻结")
    p_freeze.add_argument("--submission", type=str, required=True)
    p_freeze.add_argument("--tag", type=str, default="v1.0-final")
    p_freeze.add_argument("--output", type=str, default=None)

    p_release = subparsers.add_parser("check-release", help="发布包校验")
    p_release.add_argument("--bundle", type=str, required=True)
    p_release.add_argument("--output", type=str, default=None)

    args, remaining = parser.parse_known_args(argv)

    if args.command == "evaluate":
        return evaluate_cli(argv[1:])
    elif args.command == "batch":
        return batch_score_cli(argv[1:])
    elif args.command == "check":
        return check_submission_cli(argv[1:])
    elif args.command == "freeze":
        return freeze_submission_cli(argv[1:])
    elif args.command == "check-release":
        return check_release_cli(argv[1:])
    return 2


if __name__ == "__main__":
    sys.exit(main())
