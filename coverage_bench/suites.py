"""评测套件 suite YAML 的数据结构定义与加载：组、用例及套件哈希。"""
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml

from coverage_bench.config import TaskConfig, UniqueSafeLoader
from coverage_bench.errors import ConfigurationError, CoverageError, VersionMismatchError
from coverage_bench.release import ReleaseBundle


@dataclass(frozen=True)
class ScenarioCase:
    """一个场景用例：任务配置 + 场景随机种子。"""
    case_id: str
    group_id: str
    task_config: TaskConfig
    scenario_seed: int


@dataclass(frozen=True)
class SuiteGroup:
    """套件内一个评分组：同组用例共用 policy_repeats。"""
    group_id: str
    cases: List[ScenarioCase]
    policy_repeats: int


@dataclass(frozen=True)
class EvaluationSuite:
    """完整评测套件：组列表 + 版本/可见性元数据 + 套件哈希。"""
    suite_schema_version: str
    suite_id: str
    visibility: str
    task_version: str
    score_version: str
    groups: List[SuiteGroup]
    schedule_id: str
    suite_hash: str = ""

    def __post_init__(self):
        if not self.suite_hash:
            computed = hashlib.sha256(f"{self.suite_id}:{self.schedule_id}".encode("utf-8")).hexdigest()
            object.__setattr__(self, "suite_hash", computed)


def load_suite(path: Path, release: ReleaseBundle | None = None) -> EvaluationSuite:
    """从 YAML 文件加载评测套件并校验键集合、用例唯一性与版本匹配。"""
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
    text = path.read_text(encoding="utf-8")
    if "!!python" in text:
        raise ConfigurationError("Unsafe Python tags in suite YAML", code="CONFIG_INVALID")
    data = yaml.load(text, Loader=UniqueSafeLoader)
    if not isinstance(data, dict):
        raise ConfigurationError("Suite YAML root must be a mapping", code="CONFIG_INVALID")

    allowed_suite_keys = {
        "suite_schema_version", "suite_id", "visibility", "task_version",
        "score_version", "groups", "schedule_id",
    }
    if set(data.keys()) != allowed_suite_keys:
        raise ConfigurationError(f"Unexpected keys in suite: {set(data.keys()) - allowed_suite_keys}", code="CONFIG_INVALID")

    if data["task_version"] != release.protocol.task_version:
        raise VersionMismatchError(
            f"Suite task_version {data['task_version']} does not match release {release.protocol.task_version}",
            code="VERSION_MISMATCH",
            field="task_version",
        )

    all_case_ids = set()
    groups = []

    for grp_data in data["groups"]:
        allowed_grp_keys = {"group_id", "cases", "policy_repeats"}
        if set(grp_data.keys()) != allowed_grp_keys:
            raise ConfigurationError("Unexpected keys in group", code="CONFIG_INVALID")

        grp_id = grp_data["group_id"]
        repeats = grp_data["policy_repeats"]
        if type(repeats) is not int or repeats <= 0:
            raise ConfigurationError("policy_repeats must be a positive integer", code="CONFIG_INVALID")

        cases = []
        for case_data in grp_data["cases"]:
            allowed_case_keys = {"case_id", "group_id", "task_config", "scenario_seed"}
            if set(case_data.keys()) != allowed_case_keys:
                raise ConfigurationError("Unexpected keys in case", code="CONFIG_INVALID")

            cid = case_data["case_id"]
            if cid in all_case_ids:
                raise ConfigurationError(f"Duplicate case_id: {cid}", code="CONFIG_INVALID", field="case_id")
            all_case_ids.add(cid)

            if case_data["group_id"] != grp_id:
                raise ConfigurationError(f"case group_id {case_data['group_id']} != group {grp_id}", code="CONFIG_INVALID")

            seed = case_data["scenario_seed"]
            if type(seed) is not int or seed < 0 or seed >= 2**64:
                raise ConfigurationError(f"Invalid scenario_seed: {seed}", code="CONFIG_INVALID")

            tc_data = case_data["task_config"]
            if isinstance(tc_data, TaskConfig):
                tc = tc_data
            else:
                tc = TaskConfig(**tc_data)

            cases.append(ScenarioCase(
                case_id=cid,
                group_id=grp_id,
                task_config=tc,
                scenario_seed=seed,
            ))

        groups.append(SuiteGroup(
            group_id=grp_id,
            cases=cases,
            policy_repeats=repeats,
        ))

    suite_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return EvaluationSuite(
        suite_schema_version=data["suite_schema_version"],
        suite_id=data["suite_id"],
        visibility=data["visibility"],
        task_version=data["task_version"],
        score_version=data["score_version"],
        groups=groups,
        schedule_id=data["schedule_id"],
        suite_hash=suite_hash,
    )
