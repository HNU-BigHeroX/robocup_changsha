"""阶段级 deadline：基于 SIGALRM 的可嵌套超时上下文（load/act/episode 三阶段）。"""
import signal
import time
from contextlib import contextmanager
from typing import Iterator

from coverage_bench.errors import CoverageError

PHASE_LOAD = "load"
PHASE_ACT = "act"
PHASE_EPISODE = "episode"

PHASE_ERROR_CODES = {
    PHASE_LOAD: "LOAD_TIMEOUT",
    PHASE_ACT: "ACT_TIMEOUT",
    PHASE_EPISODE: "EPISODE_TIMEOUT",
}


def deadlines_supported() -> bool:
    """当前平台是否支持 setitimer/SIGALRM（仅 Linux 容器内可用）。"""
    return (
        hasattr(signal, "setitimer")
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "ITIMER_REAL")
    )


class PhaseDeadlineExceeded(CoverageError):
    """阶段超时异常：错误码按阶段取三码，附带 agent/step 定位。"""
    def __init__(
        self,
        phase: str,
        limit_ms: int,
        agent_index: int | None = None,
        step_index: int | None = None,
    ):
        code = PHASE_ERROR_CODES.get(phase, "DEADLINE_EXCEEDED")
        where = []
        if agent_index is not None:
            where.append(f"机器人 {agent_index}")
        if step_index is not None:
            where.append(f"第 {step_index} 步")
        suffix = f"（{'，'.join(where)}）" if where else ""
        super().__init__(
            f"{phase} 阶段超出 {limit_ms}ms 限额{suffix}", code=code,
        )
        self.phase = phase
        self.limit_ms = limit_ms
        self.agent_index = agent_index
        self.step_index = step_index


@contextmanager
def phase_deadline(
    limit_ms: int,
    phase: str,
    agent_index: int | None = None,
    step_index: int | None = None,
) -> Iterator[None]:
    """在指定阶段限额内执行代码块，超时抛 PhaseDeadlineExceeded；支持嵌套计时器。"""
    if phase not in PHASE_ERROR_CODES:
        raise CoverageError(f"未定义的阶段名: {phase}", code="DEADLINE_PHASE_INVALID")
    if type(limit_ms) is not int or limit_ms <= 0:
        raise CoverageError(f"阶段限额必须为正整数毫秒: {limit_ms}", code="DEADLINE_LIMIT_INVALID")
    if not deadlines_supported():
        raise CoverageError(
            "当前平台不支持阶段级超时，评测必须在 Linux 容器内运行",
            code="DEADLINE_UNAVAILABLE",
        )

    def _on_alarm(_signum: int, _frame: object) -> None:
        raise PhaseDeadlineExceeded(phase, limit_ms, agent_index, step_index)

    # 进入时检查是否已有活跃外层计时器（getitimer 返回非 (0.0, 0.0) 即活跃；
    # 返回 (剩余秒数, 间隔)，即 value 在前 interval 在后，解包顺序不可颠倒）
    outer_remaining, outer_interval = signal.getitimer(signal.ITIMER_REAL)
    has_outer = not (outer_interval == 0.0 and outer_remaining == 0.0)
    if has_outer:
        # 有效限额取 min(自身限额, 外层剩余秒数)
        effective_seconds = min(limit_ms / 1000.0, outer_remaining)
    else:
        effective_seconds = limit_ms / 1000.0
    entered_at = time.monotonic()

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, effective_seconds)
    try:
        yield
    finally:
        now = time.monotonic()
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
        if has_outer:
            # 嵌套时按外层剩余时间减去内层经过时间重新武装外层计时器
            rearm = outer_remaining - (now - entered_at)
            if rearm <= 0:
                # 外层预算在内层执行期间耗尽，直接调用外层处理器抛出外层阶段超时
                previous(signal.SIGALRM, None)
            else:
                signal.setitimer(signal.ITIMER_REAL, rearm)
