"""发布包管理：release.json 清单（21 组件）的结构/文件/语义校验与组件加载。"""
from dataclasses import dataclass, fields as dataclass_fields
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from coverage_bench.errors import CoverageError
from coverage_bench.protocol import ProtocolSpec, ResourceLimits

COMPONENTS = (
    "protocol", "task", "task_domain", "public_suite", "public_schedule",
    "scoring", "runtime", "official_lock", "random_baseline", "rule_baseline",
    "learning_baseline", "calibration_report", "verification_report",
    "submission_rules", "quality_checklist", "task_docs", "protocol_docs",
    "evaluation_docs", "submission_docs", "private_suite_receipt",
    "supplementary_suite_receipt",
)

MANIFEST_FIELDS = (
    "release_schema_version", "release_id", "release_state", "official_commit",
    "protocol", "score_version", "components", "protected_dependencies",
    "mpe2_source_commit", "mpe2_distribution_sha256", "created_at",
)

PROTOCOL_FIELDS = (
    "protocol_version", "task_version", "agent_capacity", "target_capacity",
    "action_dim", "position_scale", "velocity_scale", "max_episode_steps",
    "flatten_version",
)

COMPONENT_FIELDS = ("name", "path", "sha256", "size_bytes")

RESOURCE_LIMIT_FIELDS = tuple(f.name for f in dataclass_fields(ResourceLimits))

VALID_RELEASE_STATES = ("development", "frozen")

HEX_DIGITS = set("0123456789abcdefABCDEF")
LOWER_HEX_DIGITS = set("0123456789abcdef")


@dataclass(frozen=True)
class ReleaseComponent:
    """发布清单中的单个组件条目（名称、相对路径、SHA-256、大小）。"""
    name: str
    path: str
    sha256: str
    size_bytes: int = 0


@dataclass(frozen=True)
class ReleaseBundle:
    """通过校验的发布包：清单字段 + 协议 + 组件集合，附清单来源路径与哈希。"""
    release_schema_version: str
    release_id: str
    release_state: str
    official_commit: Optional[str]
    protocol: ProtocolSpec
    score_version: str
    components: Tuple[Any, ...]
    protected_dependencies: Dict[str, str]
    mpe2_source_commit: Optional[str]
    mpe2_distribution_sha256: Optional[str]
    created_at: str
    manifest_path: Optional[Path] = None
    manifest_sha256: Optional[str] = None


@dataclass(frozen=True)
class ReleaseValidationCheck:
    """单项校验结果。"""
    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class ReleaseValidationReport:
    """发布包整体校验报告：valid 仅当全部检查通过。"""
    valid: bool
    checks: Tuple[ReleaseValidationCheck, ...]


def _is_hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in HEX_DIGITS for c in value)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _is_rfc3339(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _is_plain_int(value: Any) -> bool:
    return type(value) is int


def _structural_checks(data: Dict[str, Any]) -> List[ReleaseValidationCheck]:
    checks: List[ReleaseValidationCheck] = []

    unknown = set(data.keys()) - set(MANIFEST_FIELDS)
    missing = set(MANIFEST_FIELDS) - set(data.keys())
    checks.append(ReleaseValidationCheck(
        name="manifest_fields",
        passed=not unknown and not missing,
        message="发布清单字段集合与契约一致" if not unknown and not missing
        else f"发布清单字段不符，未知 {sorted(unknown)}，缺失 {sorted(missing)}",
    ))

    schema_ver = data.get("release_schema_version")
    checks.append(ReleaseValidationCheck(
        name="release_schema_version",
        passed=schema_ver == "coverage-release/1.0",
        message="发布包 schema 必须为 coverage-release/1.0" if schema_ver != "coverage-release/1.0" else "通过",
    ))

    rel_id = data.get("release_id")
    checks.append(ReleaseValidationCheck(
        name="release_id",
        passed=bool(isinstance(rel_id, str) and rel_id.strip()),
        message="release_id 必须为非空字符串" if not isinstance(rel_id, str) or not rel_id.strip() else "通过",
    ))

    state = data.get("release_state")
    if state not in VALID_RELEASE_STATES:
        checks.append(ReleaseValidationCheck(
            name="release_state", passed=False,
            message=f"发布状态必须为 {' 或 '.join(VALID_RELEASE_STATES)}",
        ))
        state = None
    else:
        checks.append(ReleaseValidationCheck(name="release_state", passed=True, message="通过"))

    frozen = state == "frozen"

    off_commit = data.get("official_commit")
    if off_commit is None:
        commit_ok = not frozen
        commit_message = "frozen 发布包必须提供 official_commit" if frozen else "通过"
    else:
        commit_ok = _is_hex(off_commit, 40)
        commit_message = "official_commit 必须为 40 位十六进制 Git 提交哈希"
    checks.append(ReleaseValidationCheck(
        name="official_commit", passed=commit_ok,
        message="通过" if commit_ok else commit_message,
    ))

    mpe2_commit = data.get("mpe2_source_commit")
    if mpe2_commit is None:
        mpe2_ok = not frozen
        mpe2_message = "frozen 发布包必须提供 mpe2_source_commit"
    else:
        mpe2_ok = _is_hex(mpe2_commit, 40)
        mpe2_message = "mpe2_source_commit 必须为 40 位十六进制 Git 提交哈希"
    checks.append(ReleaseValidationCheck(
        name="mpe2_source_commit", passed=mpe2_ok,
        message="通过" if mpe2_ok else mpe2_message,
    ))

    mpe2_dist = data.get("mpe2_distribution_sha256")
    if mpe2_dist is None:
        dist_ok = not frozen
        dist_message = "frozen 发布包必须提供 mpe2_distribution_sha256"
    else:
        dist_ok = _is_hex(mpe2_dist, 64)
        dist_message = "mpe2_distribution_sha256 必须为 64 位十六进制哈希"
    checks.append(ReleaseValidationCheck(
        name="mpe2_distribution_sha256", passed=dist_ok,
        message="通过" if dist_ok else dist_message,
    ))

    score_version = data.get("score_version")
    if score_version is None:
        score_ok = not frozen
        score_message = "frozen 发布包必须提供 score_version"
    else:
        score_ok = isinstance(score_version, str) and bool(score_version.strip())
        score_message = "score_version 必须为非空字符串"
    checks.append(ReleaseValidationCheck(
        name="score_version", passed=score_ok,
        message="通过" if score_ok else score_message,
    ))

    created_at = data.get("created_at")
    checks.append(ReleaseValidationCheck(
        name="created_at",
        passed=_is_rfc3339(created_at),
        message="created_at 必须为带时区的 RFC3339 时间戳",
    ))

    protocol = data.get("protocol")
    if not isinstance(protocol, dict):
        checks.append(ReleaseValidationCheck(
            name="protocol", passed=False, message="protocol 必须为映射对象",
        ))
    else:
        if set(protocol.keys()) != set(PROTOCOL_FIELDS):
            checks.append(ReleaseValidationCheck(
                name="protocol", passed=False,
                message=f"protocol 字段集合不符，未知 {sorted(set(protocol.keys()) - set(PROTOCOL_FIELDS))}，"
                        f"缺失 {sorted(set(PROTOCOL_FIELDS) - set(protocol.keys()))}",
            ))
        else:
            typed_ok = (
                isinstance(protocol["protocol_version"], str)
                and isinstance(protocol["task_version"], str)
                and isinstance(protocol["flatten_version"], str)
                and all(_is_plain_int(protocol[key]) for key in
                        ("agent_capacity", "target_capacity", "action_dim", "max_episode_steps"))
                and all(_is_finite_number(protocol[key])
                        for key in ("position_scale", "velocity_scale"))
            )
            checks.append(ReleaseValidationCheck(
                name="protocol", passed=typed_ok,
                message="通过" if typed_ok else "protocol 字段类型不符合 ProtocolSpec",
            ))

    protected = data.get("protected_dependencies")
    protected_ok = (
        isinstance(protected, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in protected.items())
    )
    checks.append(ReleaseValidationCheck(
        name="protected_dependencies", passed=protected_ok,
        message="protected_dependencies 必须为字符串到字符串的映射",
    ))

    components = data.get("components")
    if not isinstance(components, list):
        checks.append(ReleaseValidationCheck(
            name="components", passed=False, message="components 必须为列表",
        ))
        return checks

    malformed = [
        c for c in components
        if not isinstance(c, dict)
        or set(c.keys()) != set(COMPONENT_FIELDS)
        or not isinstance(c["name"], str) or not c["name"]
        or not isinstance(c["path"], str) or not c["path"]
        or not isinstance(c["sha256"], str)
        or not _is_plain_int(c["size_bytes"])
    ]
    checks.append(ReleaseValidationCheck(
        name="components",
        passed=not malformed,
        message="components 每项必须精确包含字符串 name、字符串 path、字符串 sha256 与整数 size_bytes"
        if malformed else "通过",
    ))
    if malformed:
        return checks

    unknown_names = sorted({c["name"] for c in components} - set(COMPONENTS))
    checks.append(ReleaseValidationCheck(
        name="component_names",
        passed=not unknown_names,
        message="组件名称必须取自发布白名单" if unknown_names else "通过",
    ))

    return checks


def _validate_component_payloads(
    data: Dict[str, Any],
    base_dir: Path,
    bundle_factory,
) -> List[ReleaseValidationCheck]:
    checks: List[ReleaseValidationCheck] = []
    comp_dict = {c["name"]: c for c in data["components"]}
    protocol = ProtocolSpec(**data["protocol"])
    if "protocol" in comp_dict:
        checks.append(_check_protocol_component(comp_dict["protocol"], base_dir, protocol))

    for comp in ("task", "task_domain"):
        if comp in comp_dict:
            checks.append(_check_task_component(comp, comp_dict[comp], base_dir, protocol))

    if "scoring" in comp_dict:
        checks.append(_check_scoring_component(
            comp_dict["scoring"], base_dir, data.get("score_version"),
        ))

    if "runtime" in comp_dict:
        checks.append(_check_runtime_component(comp_dict["runtime"], base_dir))

    if "public_suite" in comp_dict:
        checks.append(_check_suite_component(
            comp_dict["public_suite"], base_dir, bundle_factory, protocol,
        ))

    if "public_schedule" in comp_dict:
        checks.append(_check_schedule_component(comp_dict["public_schedule"], base_dir))

    if "public_schedule" in comp_dict and "public_suite" in comp_dict:
        checks.append(_check_schedule_covers_suite(
            comp_dict["public_schedule"], comp_dict["public_suite"], base_dir, bundle_factory,
        ))

    return checks


def _component_file(component: Dict[str, Any], base_dir: Path) -> Path:
    return base_dir / Path(component["path"])


def _load_document(path: Path, name: str) -> Tuple[Optional[Any], Optional[str]]:
    import yaml
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"组件 {name} 无法解析为文档: {exc}"


def _check_protocol_component(component, base_dir, protocol) -> ReleaseValidationCheck:
    import yaml
    path = _component_file(component, base_dir)
    try:
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ReleaseValidationCheck("semantic_protocol", False, f"protocol 组件解析失败: {exc}")
    expected = {f.name: getattr(protocol, f.name) for f in dataclass_fields(protocol)}
    ok = isinstance(body, dict) and body == expected
    return ReleaseValidationCheck(
        "semantic_protocol", ok,
        "通过" if ok else "protocol 组件内容与发布清单顶层 protocol 字段不一致",
    )


def _check_task_component(name, component, base_dir, protocol) -> ReleaseValidationCheck:
    from coverage_bench.config import load_task_config, validate_task_config
    path = _component_file(component, base_dir)
    try:
        config = load_task_config(path)
        validate_task_config(config, protocol)
    except Exception as exc:
        return ReleaseValidationCheck(
            f"semantic_{name}", False, f"{name} 组件不是有效任务配置: {exc}",
        )
    return ReleaseValidationCheck(f"semantic_{name}", True, "通过")


def _check_scoring_component(component, base_dir, score_version) -> ReleaseValidationCheck:
    from coverage_bench.scoring import scoring_config_from_data
    body, error = _load_document(_component_file(component, base_dir), "scoring")
    if error is not None:
        return ReleaseValidationCheck("semantic_scoring", False, error)
    try:
        scoring = scoring_config_from_data(body, "发布包 scoring 组件")
    except CoverageError as exc:
        return ReleaseValidationCheck("semantic_scoring", False, f"scoring 组件无效: {exc.message}")
    if score_version is not None and scoring.score_version != score_version:
        return ReleaseValidationCheck(
            "semantic_scoring", False,
            f"scoring 组件 score_version 与发布清单不一致: {scoring.score_version} != {score_version}",
        )
    return ReleaseValidationCheck("semantic_scoring", True, "通过")


def _check_runtime_component(component, base_dir) -> ReleaseValidationCheck:
    body, error = _load_document(_component_file(component, base_dir), "runtime")
    if error is not None:
        return ReleaseValidationCheck("semantic_runtime", False, error)
    if not isinstance(body, dict) or set(body.keys()) != set(RESOURCE_LIMIT_FIELDS):
        return ReleaseValidationCheck(
            "semantic_runtime", False,
            f"runtime 组件字段集合必须与 ResourceLimits 一致: {sorted(RESOURCE_LIMIT_FIELDS)}",
        )
    bad = [key for key, value in body.items() if not _is_plain_int(value) or value < 0]
    if bad:
        return ReleaseValidationCheck(
            "semantic_runtime", False,
            f"runtime 组件字段必须为非负整数: {sorted(bad)}",
        )
    return ReleaseValidationCheck("semantic_runtime", True, "通过")


def _check_suite_component(component, base_dir, bundle_factory, protocol) -> ReleaseValidationCheck:
    from coverage_bench.suites import load_suite
    path = _component_file(component, base_dir)
    try:
        suite = load_suite(path, release=bundle_factory())
    except Exception as exc:
        return ReleaseValidationCheck("semantic_public_suite", False, f"public_suite 组件无效: {exc}")

    if suite.task_version != protocol.task_version:
        return ReleaseValidationCheck(
            "semantic_public_suite", False,
            f"public_suite 组件 task_version 与发布协议不一致: "
            f"{suite.task_version} != {protocol.task_version}",
        )
    return ReleaseValidationCheck("semantic_public_suite", True, "通过")


def _check_schedule_component(component, base_dir) -> ReleaseValidationCheck:
    from coverage_bench.schedules import seed_schedule_from_data
    body, error = _load_document(_component_file(component, base_dir), "public_schedule")
    if error is not None:
        return ReleaseValidationCheck("semantic_public_schedule", False, error)
    try:
        seed_schedule_from_data(body, "发布包 public_schedule 组件")
    except CoverageError as exc:
        return ReleaseValidationCheck(
            "semantic_public_schedule", False, f"public_schedule 组件无效: {exc.message}",
        )
    return ReleaseValidationCheck("semantic_public_schedule", True, "通过")


def _check_schedule_covers_suite(
    schedule_component, suite_component, base_dir, bundle_factory,
) -> ReleaseValidationCheck:
    from coverage_bench.schedules import seed_schedule_from_data
    from coverage_bench.suites import load_suite

    name = "semantic_schedule_covers_suite"
    try:
        suite = load_suite(_component_file(suite_component, base_dir), release=bundle_factory())
    except Exception as exc:
        return ReleaseValidationCheck(name, False, f"public_suite 组件无效: {exc}")

    body, error = _load_document(_component_file(schedule_component, base_dir), "public_schedule")
    if error is not None:
        return ReleaseValidationCheck(name, False, error)
    try:
        schedule = seed_schedule_from_data(body, "发布包 public_schedule 组件")
    except CoverageError as exc:
        return ReleaseValidationCheck(name, False, f"public_schedule 组件无效: {exc.message}")

    for group in suite.groups:
        for case in group.cases:
            for repeat in range(group.policy_repeats):
                indices = sorted(r.agent_index for r in schedule.get_records(case.case_id, repeat))
                expected = list(range(case.task_config.num_agents))
                if indices != expected:
                    return ReleaseValidationCheck(
                        name, False,
                        f"公开安排未按套件覆盖用例 {case.case_id} 的第 {repeat} 次重复: "
                        f"期望机器人编号 {expected}，实际 {indices}",
                    )
    return ReleaseValidationCheck(name, True, "通过")


def _component_file_checks(data: Dict[str, Any], base_dir: Path) -> List[ReleaseValidationCheck]:
    checks: List[ReleaseValidationCheck] = []
    components = data["components"]

    seen_names = set()
    seen_paths = set()
    for c in components:
        name = c["name"]
        if name in seen_names:
            checks.append(ReleaseValidationCheck(
                name=f"duplicate_{name}", passed=False, message=f"组件名称重复: {name}",
            ))
        seen_names.add(name)

        normalized = Path(c["path"]).as_posix().lower()
        if normalized in seen_paths:
            checks.append(ReleaseValidationCheck(
                name=f"duplicate_path_{name}", passed=False, message=f"组件路径重复: {c['path']}",
            ))
        seen_paths.add(normalized)

    comp_dict = {c["name"]: c for c in components}
    frozen = data.get("release_state") == "frozen"

    for comp in COMPONENTS:
        if comp not in comp_dict:
            if frozen:
                checks.append(ReleaseValidationCheck(
                    name=f"component_{comp}", passed=False, message=f"缺少必需组件: {comp}",
                ))
            continue

        c_info = comp_dict[comp]
        c_path_str = c_info["path"]
        c_size = c_info["size_bytes"]
        c_sha = c_info["sha256"]

        if not isinstance(c_path_str, str) or not c_path_str:
            checks.append(ReleaseValidationCheck(
                name=f"path_missing_{comp}", passed=False, message=f"组件缺少有效路径: {comp}",
            ))
            continue

        rel_path = Path(c_path_str)
        if rel_path.is_absolute() or ".." in rel_path.parts or ":" in c_path_str:
            checks.append(ReleaseValidationCheck(
                name=f"path_security_{comp}", passed=False,
                message=f"组件路径包含非法绝对路径或跨目录穿越: {c_path_str}",
            ))
            continue

        if not _is_plain_int(c_size) or c_size < 0:
            checks.append(ReleaseValidationCheck(
                name=f"size_missing_{comp}", passed=False, message=f"组件缺少有效大小: {comp}",
            ))
            continue

        if not _is_hex(c_sha, 64) or not all(ch in LOWER_HEX_DIGITS for ch in c_sha):
            checks.append(ReleaseValidationCheck(
                name=f"sha256_missing_{comp}", passed=False,
                message=f"组件缺少有效 SHA-256 摘要: {comp}",
            ))
            continue

        c_path = base_dir / rel_path
        if not c_path.is_file():
            checks.append(ReleaseValidationCheck(
                name=f"file_exists_{comp}", passed=False, message=f"组件文件不存在: {c_path}",
            ))
            continue

        act_size = c_path.stat().st_size
        if c_size != act_size:
            checks.append(ReleaseValidationCheck(
                name=f"size_match_{comp}", passed=False,
                message=f"组件大小不匹配: {comp} (期望 {c_size}, 实际 {act_size})",
            ))
            continue

        act_hash = hashlib.sha256(c_path.read_bytes()).hexdigest()
        if c_sha != act_hash:
            checks.append(ReleaseValidationCheck(
                name=f"sha256_match_{comp}", passed=False, message=f"组件哈希不匹配: {comp}",
            ))
            continue

        checks.append(ReleaseValidationCheck(
            name=f"component_{comp}", passed=True, message="组件有效并通过校验",
        ))

    return checks


def _build_bundle(data: Dict[str, Any], manifest_path: Path) -> ReleaseBundle:
    components_list = tuple(
        ReleaseComponent(
            name=c["name"], path=c["path"], sha256=c["sha256"], size_bytes=c["size_bytes"],
        )
        for c in data["components"]
    )
    protocol_data = data["protocol"]
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ReleaseBundle(
        release_schema_version=data["release_schema_version"],
        release_id=data["release_id"],
        release_state=data["release_state"],
        official_commit=data["official_commit"],
        protocol=ProtocolSpec(**protocol_data),
        score_version=data["score_version"] or "coverage-score/1.0",
        components=components_list,
        protected_dependencies=dict(data["protected_dependencies"]),
        mpe2_source_commit=data["mpe2_source_commit"],
        mpe2_distribution_sha256=data["mpe2_distribution_sha256"],
        created_at=data["created_at"],
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )


def validate_release(path: Path) -> ReleaseValidationReport:
    """三段式校验发布清单：结构检查 → 组件文件哈希 → 组件载荷语义检查。"""
    target = Path(path)
    if not target.is_file():
        return ReleaseValidationReport(
            valid=False,
            checks=(ReleaseValidationCheck(name="file_exists", passed=False, message="发布包文件不存在"),),
        )

    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return ReleaseValidationReport(
            valid=False,
            checks=(ReleaseValidationCheck(name="valid_json", passed=False, message=f"JSON 解析失败: {exc}"),),
        )

    if not isinstance(data, dict):
        return ReleaseValidationReport(
            valid=False,
            checks=(ReleaseValidationCheck(
                name="manifest_root", passed=False, message="发布清单根结构必须为对象",
            ),),
        )

    checks = _structural_checks(data)
    if not all(c.passed for c in checks):
        return ReleaseValidationReport(valid=False, checks=tuple(checks))

    checks.extend(_component_file_checks(data, target.parent))

    def bundle_factory() -> ReleaseBundle:
        return _build_bundle(data, target)

    checks.extend(_validate_component_payloads(data, target.parent, bundle_factory))

    valid = all(c.passed for c in checks)
    return ReleaseValidationReport(valid=valid, checks=tuple(checks))


def load_release(path: Path) -> ReleaseBundle:
    """校验并加载发布包，校验失败即抛 CoverageError。"""
    target = Path(path).resolve()
    report = validate_release(target)
    if not report.valid:
        failed_msgs = [c.message for c in report.checks if not c.passed]
        raise CoverageError(
            f"发布包校验失败: {'; '.join(failed_msgs)}", code="RELEASE_INVALID",
        )
    data = json.loads(target.read_text(encoding="utf-8"))
    return _build_bundle(data, target)


def component_path(bundle: ReleaseBundle, name: str) -> Path:
    """解析指定组件在宿主上的绝对路径，组件缺失或身份未知即抛错。"""
    if bundle.manifest_path is None:
        raise CoverageError(
            f"发布包身份不可确定，无法解析组件路径: {name}", code="RELEASE_IDENTITY_UNKNOWN",
        )
    for component in bundle.components:
        if component.name == name:
            if not isinstance(component.path, str) or not component.path:
                break
            return Path(bundle.manifest_path).parent / component.path
    raise CoverageError(f"发布包缺少必需组件: {name}", code="RELEASE_COMPONENT_MISSING")


def compute_release_bundle_hash(bundle: ReleaseBundle) -> str:
    """返回发布清单的规范化 SHA-256，作为发布包身份。"""
    if bundle.manifest_sha256 is None:
        raise CoverageError(
            "发布包身份不可确定，无法计算 release_bundle_hash", code="RELEASE_IDENTITY_UNKNOWN",
        )
    return bundle.manifest_sha256


def load_release_scoring(bundle: ReleaseBundle):
    """从发布包 scoring 组件加载评分配置。"""
    from coverage_bench.scoring import scoring_config_from_data
    path = component_path(bundle, "scoring")
    body, error = _load_document(path, "scoring")
    if error is not None:
        raise CoverageError(error, code="RELEASE_COMPONENT_INVALID")
    return scoring_config_from_data(body, str(path))


def load_release_runtime_limits(bundle: ReleaseBundle) -> ResourceLimits:
    """从发布包 runtime 组件加载资源限额。"""
    path = component_path(bundle, "runtime")
    body, error = _load_document(path, "runtime")
    if error is not None:
        raise CoverageError(error, code="RELEASE_COMPONENT_INVALID")
    if not isinstance(body, dict) or set(body.keys()) != set(RESOURCE_LIMIT_FIELDS):
        raise CoverageError(
            f"runtime 组件字段集合必须与 ResourceLimits 一致: {path}", code="RELEASE_COMPONENT_INVALID",
        )
    return ResourceLimits(**body)


def load_release_schedule(bundle: ReleaseBundle):
    """从发布包 public_schedule 组件加载种子安排。"""
    from coverage_bench.schedules import seed_schedule_from_data
    path = component_path(bundle, "public_schedule")
    body, error = _load_document(path, "public_schedule")
    if error is not None:
        raise CoverageError(error, code="RELEASE_COMPONENT_INVALID")
    return seed_schedule_from_data(body, str(path))
