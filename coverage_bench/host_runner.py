"""宿主侧容器执行器：构造加固的 docker run 命令并代跑评测容器。"""
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from coverage_bench.errors import CoverageError
from coverage_bench.protocol import ResourceLimits

# 容器内使用的非 root uid/gid，避免以 root 身份运行选手代码
NON_ROOT_UID = 10001
NON_ROOT_GID = 10001

# docker 客户端自身清理命令的时限（秒）
_CLEANUP_COMMAND_TIMEOUT_S = 30.0

# 宿主轮询容器退出状态的间隔（秒）
_POLL_INTERVAL_S = 0.05

# 提交挂载点基路径：容器内实际挂载为 /submissions/<participant_id>。
# 不能用固定路径：validate_submission 强制清单 participant_id 与提交目录
# basename 相等，固定挂载点会使容器内 basename 恒为常量而令全体参赛者
# 在校验阶段以退出码 2 失败（score_all 记 JUDGE_CONFIG_INVALID）
_SUBMISSIONS_BASE = "/submissions"
_SUITE_MOUNT = "/judge/private-suite.yaml"
_SEEDS_MOUNT = "/judge/private-seeds.yaml"
_OUTPUT_MOUNT = "/output"

# docker run 在退出码 125/126/127 表示 docker 自身或镜像层面失败，
# 而非容器内进程的退出码，score_all 据此区分基础设施失败
INFRASTRUCTURE_EXIT_CODES = frozenset({125, 126, 127})


@dataclass(frozen=True, slots=True)
class ContainerOutcome:
    """单次容器运行的最终退出码、是否超时与宿主侧日志路径。"""
    returncode: int
    timed_out: bool
    log_path: Path


def _docker_executable() -> str:
    # 优先按 PATH（含 PATHEXT）解析 docker 可执行文件，找不到时保留原始名称快速失败
    resolved = shutil.which("docker")
    if resolved is None:
        return "docker"
    return resolved


def container_name(participant_id: str, run_id: str) -> str:
    """按参赛者编号与 run_id 推导容器名。"""
    return f"coverage-eval-{participant_id}-{run_id}"


def cleanup_commands(name: str) -> list[list[str]]:
    """按顺序生成容器的 docker kill / rm -f 清理命令。"""
    docker = _docker_executable()
    return [
        [docker, "kill", name],
        [docker, "rm", "-f", name],
    ]


def force_cleanup(name: str) -> None:
    """强制清理卡死的容器，两条清理命令全部失败时按基础设施异常快速失败。"""
    # 清理不依赖卡死的容器自行完成：先 docker kill，失败再 docker rm -f，
    # 两条命令自身都带 30 秒上限，全部失败按基础设施异常快速失败
    commands = cleanup_commands(name)
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=_CLEANUP_COMMAND_TIMEOUT_S,
            )
            if completed.returncode == 0:
                return
        except (subprocess.TimeoutExpired, OSError):
            pass
    raise CoverageError(
        f"容器强制清理失败（kill 与 rm -f 均未成功）: {name}",
        code="CONTAINER_CLEANUP_FAILED",
    )


def build_run_command(
    image_ref: str,
    submission_dir: Path,
    suite_path: Path,
    seeds_path: Path,
    output_dir: Path,
    limits: ResourceLimits,
    name: str | None = None,
) -> list[str]:
    """构造加固 docker run 命令：网络全关、只读根文件系统、非 root、限额与四组挂载。"""
    # 参赛者编号取提交目录名：validate_submission 强制清单 participant_id 与
    # 目录名相等，且编号受 PARTICIPANT_ID_REGEX (^P\d{3}$) 约束、字符集受控，
    # 可安全拼入容器内挂载路径；目录名不合规的提交会由容器内同一校验以退出
    # 码 2 确定性拒绝（subprocess 以列表传参不经 shell，无注入面）
    submission_host = Path(submission_dir).resolve()
    participant_id = submission_host.name
    submission_mount = f"{_SUBMISSIONS_BASE}/{participant_id}"
    # 容器名缺省时从参赛者编号与随机 run_id 推导；--name 与 --rm 必须同时使用
    if name is None:
        import uuid

        name = container_name(participant_id, uuid.uuid4().hex[:12])
    suite_host = Path(suite_path).resolve()
    seeds_host = Path(seeds_path).resolve()
    output_host = Path(output_dir).resolve()

    return [
        _docker_executable(),
        "run",
        "--rm",
        "--name", name,
        # 网络全关，杜绝容器访问外部主机或互联网
        "--network", "none",
        # 根文件系统只读，仅 /tmp 与输出挂载点可写
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        # 非 root 用户运行
        "--user", f"{NON_ROOT_UID}:{NON_ROOT_GID}",
        # 进程数与内存上限沿用 worker 资源限额
        "--pids-limit", str(limits.max_processes_per_worker),
        "--memory", f"{limits.memory_mib_per_worker}m",
        # 只读根文件系统下 Python 与 NumPy 需要可写临时目录；提交快照
        # （COVERAGE_WORK_DIR 指向 /tmp 下，见 Dockerfile ENV）同样驻留于此：
        # 快照先整目录复制后才校验 submission_bytes_total，且整个评测生命周期
        # 存活（entry 模块与产物均从快照加载），tmpfs 上限须覆盖两者之和
        "--tmpfs",
        f"/tmp:rw,size={limits.scratch_bytes_per_worker + limits.submission_bytes_total}",
        # 三个只读输入挂载 + 一个可写输出挂载；提交挂载点含参赛者编号，
        # 保证容器内提交目录 basename 与清单 participant_id 一致
        "-v", f"{submission_host}:{submission_mount}:ro",
        "-v", f"{suite_host}:{_SUITE_MOUNT}:ro",
        "-v", f"{seeds_host}:{_SEEDS_MOUNT}:ro",
        "-v", f"{output_host}:{_OUTPUT_MOUNT}:rw",
        image_ref,
        "python", "scripts/evaluate_one.py",
        "--submission", submission_mount,
        "--suite", _SUITE_MOUNT,
        "--seeds", _SEEDS_MOUNT,
        "--output", _OUTPUT_MOUNT,
    ]


def run_container(
    command: list[str],
    name: str,
    timeout_ms: int,
    log_path: Path,
) -> ContainerOutcome:
    """启动容器并轮询至退出或超时；超时则强制清理，日志合流写入 log_path。"""
    # stdout 与 stderr 合流写入宿主侧日志文件
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    timed_out = False
    with log_file.open("wb") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        # 轮询 poll() 直到容器退出或超出整场 hard deadline
        while True:
            returncode = process.poll()
            if returncode is not None:
                break
            elapsed_ms = (time.monotonic() - start) * 1000.0
            if elapsed_ms >= timeout_ms:
                timed_out = True
                force_cleanup(name)
                break
            time.sleep(_POLL_INTERVAL_S)
        # 强制清理后 docker run 进程会随之退出，等待其收敛以便取得最终退出码
        try:
            returncode = process.wait(timeout=_CLEANUP_COMMAND_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait()

    return ContainerOutcome(returncode=returncode, timed_out=timed_out, log_path=log_file)
