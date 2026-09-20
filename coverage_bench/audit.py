"""提交静态审计：按后缀白/黑名单检查产物文件，并解析 Python 源码拒绝危险调用。

本模块只在容器评测前的静态阶段运行，不做任何动态执行。
"""
import ast
import os
from dataclasses import dataclass, field
from pathlib import Path

from coverage_bench.errors import CoverageError

# 产物后缀白名单：artifacts 目录下仅允许登记这些后缀的文件
ALLOWED_ARTIFACT_SUFFIXES = frozenset({
    ".safetensors", ".npy", ".npz", ".json", ".yaml", ".yml", ".txt", ".csv",
})

# 产物后缀黑名单：pickle 系序列化载体，存在任意代码执行的反序列化风险
FORBIDDEN_ARTIFACT_SUFFIXES = frozenset({
    ".pkl", ".pickle", ".pt", ".pth", ".joblib", ".dill",
})

# 原生二进制后缀：提交目录任意深度都禁止出现
FORBIDDEN_NATIVE_SUFFIXES = frozenset({
    ".so", ".pyd", ".dll", ".dylib", ".exe",
})

# 扫描符号清单：命中只写入 scan_hits 供人工审核，不参与拒绝判断
# torch 在此仅承担人工审核记录职责，torch.load 的硬拒绝见下方反序列化调用集合
SCAN_SYMBOLS = (
    "subprocess", "os.system", "os.popen", "socket", "urllib", "requests",
    "pickle", "dill", "cloudpickle", "joblib", "eval", "exec", "ctypes", "torch",
)

# pickle 反序列化调用：硬拒绝
# torch.load 本质是 pickle 反序列化且不校验扩展名，白名单内任意后缀可携带
# pickle 载荷经其执行，故与 pickle 系同列（设计 §6 AT-AUD-02）
_PICKLE_DESERIALIZE_CALLS = frozenset({
    "pickle.loads", "pickle.load", "pickle.Unpickler",
    "dill.loads", "dill.load", "dill.Unpickler",
    "cloudpickle.loads", "cloudpickle.load",
    "torch.load",
})

# 外部下载调用：硬拒绝（提交运行期不允许拉取外部资源）
_DOWNLOAD_CALLS = frozenset({
    "urllib.request.urlopen", "urllib.request.urlretrieve",
    "requests.get", "requests.post", "requests.put", "requests.delete",
    "requests.head", "requests.options", "requests.patch", "requests.request",
})

# 未登记动态载入调用：硬拒绝（当前提交契约不存在任何动态载入登记机制）
_DYNAMIC_LOAD_CALLS = frozenset({
    "importlib.import_module", "importlib.reload",
    "importlib.util.spec_from_file_location", "__import__",
})


@dataclass(frozen=True)
class AuditFinding:
    """单条审计发现（拒绝项或扫描命中）。"""
    code: str
    path: str
    line: int | None
    symbol: str | None
    detail: str


@dataclass(frozen=True)
class AuditReport:
    """审计结果：rejections 任一非空即拒绝提交，scan_hits 仅供人工审核。"""
    rejections: tuple[AuditFinding, ...] = ()
    scan_hits: tuple[AuditFinding, ...] = ()
    review: dict[str, str] = field(default_factory=dict)


def _collect_bindings(tree: ast.AST) -> dict[str, str]:
    # 收集 import 绑定，把本地名字映射到完整点分路径，供调用点解析使用
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bindings[alias.asname] = alias.name
                else:
                    # import a.b 只绑定根名 a，属性访问 a.b.c 由解析器逐段拼接
                    root_name = alias.name.split(".")[0]
                    bindings.setdefault(root_name, root_name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    bindings[local_name] = f"{node.module}.{alias.name}"
    return bindings


def _resolve_dotted(node: ast.expr, bindings: dict[str, str]) -> str | None:
    # 把 Name/Attribute 链解析为点分路径；非纯名字链（如函数调用结果）返回 None
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    parts.reverse()
    head, rest = parts[0], parts[1:]
    if head in bindings:
        dotted = bindings[head]
        if rest:
            dotted = f"{dotted}.{'.'.join(rest)}"
        return dotted
    return ".".join(parts)


def _match_scan_symbol(dotted: str) -> bool:
    # 仅做从头对齐的段边界匹配，避免 self.pickle 之类的路径误报
    return any(dotted == symbol or dotted.startswith(symbol + ".") for symbol in SCAN_SYMBOLS)


def _add_hit(
    hits: list[AuditFinding],
    seen: set[tuple[str, int, str]],
    rel_path: str,
    line: int,
    symbol: str,
) -> None:
    key = (rel_path, line, symbol)
    if key in seen:
        return
    seen.add(key)
    hits.append(
        AuditFinding(
            code="SCAN_HIT",
            path=rel_path,
            line=line,
            symbol=symbol,
            detail="源码命中扫描符号清单，仅供人工审核，不参与拒绝判断",
        )
    )


def _add_rejection(
    rejections: list[AuditFinding],
    seen: set[tuple[str, int, str, str]],
    code: str,
    rel_path: str,
    line: int,
    symbol: str,
    detail: str,
) -> None:
    key = (code, rel_path, line, symbol)
    if key in seen:
        return
    seen.add(key)
    rejections.append(
        AuditFinding(code=code, path=rel_path, line=line, symbol=symbol, detail=detail)
    )


def _analyze_python_source(
    rel_path: str,
    source_path: Path,
    rejections: list[AuditFinding],
    seen_rejections: set[tuple[str, int, str, str]],
    scan_hits: list[AuditFinding],
    seen_hits: set[tuple[str, int, str]],
) -> None:
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        # 快速失败：语法不可解析的源码无法静态审核，静默跳过会让违规代码逃过审核
        raise CoverageError(
            f"Python 源码解析失败: {rel_path}: {exc}",
            code="PYTHON_SOURCE_UNPARSABLE",
        )

    bindings = _collect_bindings(tree)
    for node in ast.walk(tree):
        # import 语句本身按导入名记录扫描命中（如 import subprocess / import urllib.request）
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _match_scan_symbol(alias.name):
                    _add_hit(scan_hits, seen_hits, rel_path, node.lineno, alias.name)
            continue
        if isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    imported = f"{node.module}.{alias.name}"
                    if _match_scan_symbol(imported):
                        _add_hit(scan_hits, seen_hits, rel_path, node.lineno, imported)
            continue
        if not isinstance(node, ast.Call):
            continue

        # 调用点解析：基于 import 绑定还原完整点分路径，注释与字符串字面量不会进入 ast
        dotted = _resolve_dotted(node.func, bindings)
        if dotted is None:
            continue
        if dotted in _PICKLE_DESERIALIZE_CALLS:
            _add_rejection(
                rejections, seen_rejections,
                "PICKLE_DESERIALIZATION", rel_path, node.lineno, dotted,
                "源码出现 pickle 系反序列化调用",
            )
        elif dotted in _DOWNLOAD_CALLS:
            _add_rejection(
                rejections, seen_rejections,
                "EXTERNAL_DOWNLOAD", rel_path, node.lineno, dotted,
                "源码出现外部下载调用",
            )
        elif dotted in _DYNAMIC_LOAD_CALLS:
            _add_rejection(
                rejections, seen_rejections,
                "UNREGISTERED_DYNAMIC_LOAD", rel_path, node.lineno, dotted,
                "源码出现未登记的动态模块载入调用",
            )
        if _match_scan_symbol(dotted):
            _add_hit(scan_hits, seen_hits, rel_path, node.lineno, dotted)


def audit_submission(root: Path) -> AuditReport:
    """遍历提交目录执行静态审计，返回拒绝项与扫描命中清单。"""
    root = Path(root).resolve()
    if not root.is_dir():
        raise CoverageError(f"提交目录不存在: {root}", code="SUBMISSION_DIR_MISSING")

    rejections: list[AuditFinding] = []
    scan_hits: list[AuditFinding] = []
    seen_rejections: set[tuple[str, int, str, str]] = set()
    seen_hits: set[tuple[str, int, str]] = set()

    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            rel_path = file_path.relative_to(root).as_posix()
            # 后缀统一按小写折叠匹配：Windows 文件系统大小写不敏感，折叠防止 .PT/.So 绕过
            suffix = file_path.suffix.lower()

            # native 后缀优先判定，覆盖提交目录任意深度（含 artifacts 内外）
            if suffix in FORBIDDEN_NATIVE_SUFFIXES:
                _add_rejection(
                    rejections, seen_rejections,
                    "NATIVE_BINARY_FORBIDDEN", rel_path, None, None,
                    "提交目录内出现原生二进制文件",
                )
                continue

            parts = rel_path.split("/")
            if parts[0] == "artifacts":
                if suffix in FORBIDDEN_ARTIFACT_SUFFIXES:
                    _add_rejection(
                        rejections, seen_rejections,
                        "ARTIFACT_FORMAT_FORBIDDEN", rel_path, None, None,
                        "artifacts 目录下出现禁止的序列化产物后缀",
                    )
                    continue
                if suffix not in ALLOWED_ARTIFACT_SUFFIXES:
                    _add_rejection(
                        rejections, seen_rejections,
                        "ARTIFACT_FORMAT_NOT_ALLOWED", rel_path, None, None,
                        "artifacts 目录下出现白名单之外的产物后缀",
                    )
                    continue

            if suffix == ".py":
                _analyze_python_source(
                    rel_path, file_path,
                    rejections, seen_rejections, scan_hits, seen_hits,
                )

    return AuditReport(
        rejections=tuple(rejections),
        scan_hits=tuple(scan_hits),
        review={},
    )
