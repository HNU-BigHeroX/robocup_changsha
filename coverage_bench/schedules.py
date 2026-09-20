"""种子调度表：每个用例每个重复每个智能体的 build/policy 随机种子及其哈希校验。"""
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Tuple

import numpy as np

from coverage_bench.errors import CoverageError
from coverage_bench.suites import EvaluationSuite


@dataclass(frozen=True)
class PolicySeedRecord:
    """单条种子记录：某次重复中某个智能体的构建/策略种子。"""
    case_id: str
    repeat_index: int
    agent_index: int
    build_seed: int
    policy_seed: int
    group_id: str | None = None


ScheduleRecord = PolicySeedRecord


@dataclass(frozen=True)
class SeedSchedule:
    """完整种子调度表（records 有序，schedule_hash 覆盖全部记录）。"""
    schedule_schema_version: str
    schedule_id: str
    schedule_hash: str
    records: Tuple[PolicySeedRecord, ...]

    def get_records(self, case_id: str, repeat_index: int) -> Tuple[PolicySeedRecord, ...]:
        return tuple(r for r in self.records if r.case_id == case_id and r.repeat_index == repeat_index)


def compute_schedule_hash(records: Tuple[PolicySeedRecord, ...]) -> str:
    """对调度记录做规范化 JSON 序列化后计算 SHA-256。"""
    serialized = json.dumps([
        {
            "case_id": r.case_id,
            "group_id": r.group_id,
            "repeat_index": r.repeat_index,
            "agent_index": r.agent_index,
            "build_seed": r.build_seed,
            "policy_seed": r.policy_seed,
        }
        for r in records
    ], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def seed_schedule_from_data(data: Any, origin: str) -> SeedSchedule:
    """从外部数据解析并校验种子调度表（结构、类型与哈希一致性）。"""
    if not isinstance(data, dict):
        raise CoverageError(f"安排根结构必须为映射: {origin}", code="SCHEDULE_INVALID")
    required = {"schedule_schema_version", "schedule_id", "schedule_hash", "records"}
    if set(data.keys()) != required:
        raise CoverageError(
            f"安排字段集合不符: {origin}，多余 {set(data.keys()) - required}，缺失 {required - set(data.keys())}",
            code="SCHEDULE_INVALID",
        )
    records_raw = data["records"]
    if not isinstance(records_raw, list) or not records_raw:
        raise CoverageError(f"安排的 records 必须为非空列表: {origin}", code="SCHEDULE_INVALID")
    records = []
    for entry in records_raw:
        if not isinstance(entry, dict) or set(entry.keys()) != {
            "case_id", "group_id", "repeat_index", "agent_index", "build_seed", "policy_seed",
        }:
            raise CoverageError(f"安排记录字段集合不符: {origin}, {entry}", code="SCHEDULE_INVALID")
        case_id = entry["case_id"]
        group_id = entry["group_id"]
        if not isinstance(case_id, str) or not isinstance(group_id, str):
            raise CoverageError(f"安排记录的 case_id 与 group_id 必须为字符串: {origin}", code="SCHEDULE_INVALID")
        seeds = {}
        for field in ("repeat_index", "agent_index", "build_seed", "policy_seed"):
            value = entry[field]
            if type(value) is not int or value < 0:
                raise CoverageError(
                    f"安排记录字段 {field} 必须为非负整数: {origin}", code="SCHEDULE_INVALID",
                )
            seeds[field] = value
        records.append(PolicySeedRecord(
            case_id=case_id,
            group_id=group_id,
            repeat_index=seeds["repeat_index"],
            agent_index=seeds["agent_index"],
            build_seed=seeds["build_seed"],
            policy_seed=seeds["policy_seed"],
        ))
    record_tuple = tuple(records)
    declared_hash = data["schedule_hash"]
    if not isinstance(declared_hash, str):
        raise CoverageError(f"安排的 schedule_hash 必须为字符串: {origin}", code="SCHEDULE_INVALID")
    actual_hash = compute_schedule_hash(record_tuple)
    if declared_hash != actual_hash:
        raise CoverageError(
            f"安排的 schedule_hash 与记录内容不符: {declared_hash} != {actual_hash}",
            code="SCHEDULE_HASH_MISMATCH",
        )
    for field in ("schedule_schema_version", "schedule_id"):
        if not isinstance(data[field], str) or not data[field]:
            raise CoverageError(f"安排字段 {field} 必须为非空字符串: {origin}", code="SCHEDULE_INVALID")
    return SeedSchedule(
        schedule_schema_version=data["schedule_schema_version"],
        schedule_id=data["schedule_id"],
        schedule_hash=declared_hash,
        records=record_tuple,
    )


def create_seed_schedule(suite: EvaluationSuite, policy_entropy: int) -> SeedSchedule:
    """按套件结构确定性生成全量种子调度表，种子两两不重复。"""
    rng = np.random.Generator(np.random.PCG64(policy_entropy))
    used_seeds = set()
    records = []

    def next_seed() -> int:
        while True:
            s = int(rng.integers(0, 2**64, dtype=np.uint64))
            if s not in used_seeds:
                used_seeds.add(s)
                return s

    for group in suite.groups:
        for case in group.cases:
            for repeat_idx in range(group.policy_repeats):
                for agent_idx in range(case.task_config.num_agents):
                    b_seed = next_seed()
                    p_seed = next_seed()
                    records.append(ScheduleRecord(
                        case_id=case.case_id,
                        group_id=group.group_id,
                        repeat_index=repeat_idx,
                        agent_index=agent_idx,
                        build_seed=b_seed,
                        policy_seed=p_seed,
                    ))

    schedule_hash = compute_schedule_hash(tuple(records))

    return SeedSchedule(
        schedule_schema_version="coverage-schedule/1.0",
        schedule_id=suite.schedule_id,
        schedule_hash=schedule_hash,
        records=tuple(records),
    )
