import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

# 引导仓库根目录进入模块搜索路径，保证 `python scripts/evaluate_one.py` 可直接运行
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from coverage_bench.deadline import deadlines_supported, phase_deadline
from coverage_bench.errors import CoverageError
from coverage_bench.evaluation import evaluate_submission
from coverage_bench.protocol import ResourceLimits
from coverage_bench.schedules import seed_schedule_from_data
from coverage_bench.scoring import default_scoring_config
from coverage_bench.submission import validate_submission
from coverage_bench.suites import load_suite


def _resolve_manifest(submission_arg: str) -> Path:
    # 容器契约传入参赛目录，内部定位 submission.yaml；直接传入清单文件亦兼容
    sub_path = Path(submission_arg).resolve()
    if sub_path.is_dir():
        return sub_path / "submission.yaml"
    return sub_path


def evaluate_one_cli(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Coverage Bench 容器内单参赛者评测入口")
    parser.add_argument("--submission", type=str, required=True, help="参赛提交目录 (含 submission.yaml)")
    parser.add_argument("--suite", type=str, required=True, help="评测套件配置文件路径")
    parser.add_argument("--seeds", type=str, required=True, help="种子安排文件路径 (coverage-schedule/1.0)")
    parser.add_argument("--output", type=str, required=True, help="评测结果输出目录")
    args = parser.parse_args(argv)

    # 快速失败：三个输入路径必须存在，否则按配置错误退出 (code 2)
    manifest_path = _resolve_manifest(args.submission)
    if not manifest_path.is_file():
        sys.stderr.write(f"错误: 提交清单文件不存在: {args.submission}\n")
        return 2

    suite_path = Path(args.suite).resolve()
    if not suite_path.is_file():
        sys.stderr.write(f"错误: 套件文件不存在: {args.suite}\n")
        return 2

    seeds_path = Path(args.seeds).resolve()
    if not seeds_path.is_file():
        sys.stderr.write(f"错误: 种子安排文件不存在: {args.seeds}\n")
        return 2

    output_dir = Path(args.output).resolve()

    try:
        submission = validate_submission(manifest_path)
    except CoverageError as e:
        sys.stderr.write(f"提交校验错误: {e}\n")
        return 2

    try:
        suite = load_suite(suite_path)
    except CoverageError as e:
        sys.stderr.write(f"套件配置错误: {e}\n")
        return 2

    # 种子安排由外部文件注入，容器内不做任何随机派生
    try:
        seeds_data = json.loads(seeds_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        sys.stderr.write(f"种子安排文件读取失败: {e}\n")
        return 2
    try:
        schedule = seed_schedule_from_data(seeds_data, origin=str(seeds_path))
    except CoverageError as e:
        sys.stderr.write(f"种子安排配置错误: {e}\n")
        return 2

    # schedule-vs-suite 覆盖预检（与 cli.evaluate_cli release 路径一致）：
    # 种子安排未覆盖套件任一 (case, repeat) 属评委侧配置错误，若放行将在
    # run_episode 中以 CoverageError 退 4，被 score_all 误判为选手 CONTAINER_FAILED
    missing = [
        (case.case_id, rep)
        for group in suite.groups for case in group.cases
        for rep in range(group.policy_repeats)
        if not schedule.get_records(case.case_id, rep)
    ]
    if missing:
        sys.stderr.write(f"种子安排未覆盖套件用例: {missing[:5]}\n")
        return 2

    scoring = default_scoring_config(suite)
    limits = ResourceLimits()

    # 阶段级超时依赖 SIGALRM，仅 Linux 可用。官方评测链路（Linux 容器）行为不变；
    # 在不支持的平台（Windows/macOS）上降级为不设阶段超时的本地预览模式，并改用
    # 非 official 的 provenance，避免预览结果被误当成正式容器成绩。
    if deadlines_supported():
        deadline_factory = phase_deadline
        provenance = "official_container"
    else:
        def _preview_deadline(*_args, **_kwargs):
            return nullcontext()
        deadline_factory = _preview_deadline
        provenance = "local_preview"
        sys.stderr.write(
            "警告: 当前平台不支持阶段级超时（load/act/episode 限额不生效），\n"
            "      本次结果为本地预览成绩（provenance=local_preview），\n"
            "      正式成绩以官方 Linux 容器评测为准。\n"
        )

    try:
        result = evaluate_submission(
            submission=submission,
            suite=suite,
            schedule=schedule,
            scoring=scoring,
            limits=limits,
            output_dir=output_dir,
            provenance=provenance,
            deadline_factory=deadline_factory,
        )
    except CoverageError as e:
        sys.stderr.write(f"评测执行致命错误: {e}\n")
        return 4

    # 退出码映射与 evaluate_cli 保持一致；
    # resource_limit_exceeded 为已删除的 worker 进程模型遗留状态，
    # 进程内 runtime 只产生 ok/load_error/timeout/protocol_error/runtime_error，核实后移除
    if result.status == "ok":
        return 0
    if result.status in ("load_error", "timeout", "protocol_error", "runtime_error"):
        return 3
    return 4


if __name__ == "__main__":
    sys.exit(evaluate_one_cli())
