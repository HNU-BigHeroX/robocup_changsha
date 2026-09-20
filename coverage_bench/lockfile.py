"""依赖锁一致性检查：导出 uv.lock 并与仓库内 requirements-official.lock 比对。"""
import shutil
import subprocess
from pathlib import Path

from coverage_bench.errors import CoverageError

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "requirements-official.lock"

EXPORT_ARGS = (
    "export", "--format", "requirements-txt", "--no-dev",
    "--frozen", "--no-emit-project",
)


def export_requirements_text() -> str:
    """用 uv 导出当前环境的生产依赖 requirements 文本，导出失败或为空即报错。"""
    uv = shutil.which("uv")
    if uv is None:
        raise CoverageError("未找到 uv，无法导出依赖锁", code="LOCK_TOOL_MISSING")
    result = subprocess.run(
        [uv, *EXPORT_ARGS],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CoverageError(
            f"uv export 失败: {result.stderr.strip()}", code="LOCK_EXPORT_FAILED",
        )
    if not result.stdout.strip():
        raise CoverageError("uv export 输出为空", code="LOCK_EXPORT_EMPTY")
    return result.stdout


def verify_lock_drift() -> None:
    """校验 requirements-official.lock 与 uv.lock 导出一致，出现漂移即报错。"""
    if not LOCK_PATH.is_file():
        raise CoverageError(f"依赖锁文件不存在: {LOCK_PATH}", code="LOCK_MISSING")
    expected = LOCK_PATH.read_text(encoding="utf-8")
    actual = export_requirements_text()
    if expected != actual:
        raise CoverageError(
            "requirements-official.lock 与 uv.lock 不一致，请重新导出",
            code="LOCK_DRIFT",
        )
