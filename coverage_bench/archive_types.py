"""冻结与提交索引的 Pydantic 模型，全部 extra="forbid"（多写字段直接拒绝）。"""
from typing import Dict, Optional
from pydantic import BaseModel, ConfigDict


class FreezeRecord(BaseModel):
    """提交冻结记录：固化参赛者、Git 提交与产物/依赖锁/清单哈希。"""
    model_config = ConfigDict(extra="forbid")

    freeze_schema_version: str = "coverage-freeze/1.0"
    participant_id: str
    repository_url: str
    final_tag: str
    source_commit: str
    code_tree_hash: str
    artifact_hashes: Dict[str, str]
    inference_lock_hash: str
    submission_manifest_hash: str
    protocol_version: str
    task_version: str
    created_at: str
    working_tree_dirty: bool


class FreezeReceipt(BaseModel):
    """组委会出具的冻结回执，内嵌原始 FreezeRecord 并附记录哈希。"""
    model_config = ConfigDict(extra="forbid")

    receipt_id: str
    received_at: str
    record_hash: str
    freeze_record: FreezeRecord


class SubmissionIndex(BaseModel):
    """提交索引条目：汇总冻结信息与评测报告路径。"""
    model_config = ConfigDict(extra="forbid")

    index_schema_version: str = "coverage-submission-index/1.0"
    participant_id: str
    repository_url: str
    final_tag: str
    frozen_commit: str
    freeze_receipt_id: str
    artifact_hashes: Dict[str, str]
    report_path: str
    official_result_id: Optional[str] = None
