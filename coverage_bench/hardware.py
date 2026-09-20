"""默认硬件档案构造：宿主侧补写结果与容器内评测入口共用的唯一实现。"""
import os
import sys

from coverage_bench.results import HardwareProfile

# 硬件档案默认构造：宿主侧 score_all 补写结果与容器内评测入口共用的唯一实现。
# 本模块必须保持轻量：仅依赖 results 的 HardwareProfile 类型与标准库，
# 禁止导入 runtime/envs 链（pettingzoo/gymnasium/scipy 属容器内环境栈，
# 宿主侧脚本 score_all/check_submission/compare_hosts 的 import 闭包不得触达；
# 历史上该函数位于 evaluation.py，宿主仅为取默认硬件档案就被迫拖入整个
# 环境栈导致宿主解释器 ModuleNotFoundError，Task 14 移出解耦）


def _default_hardware_profile() -> HardwareProfile:
    # 无官方硬件清单时的开发档案：profile_id 与 lock 哈希使用占位值，
    # 官方评测必须通过 evaluate_submission 的 hardware_profile 参数显式传入
    return HardwareProfile(
        profile_id="host-local",
        os_image=f"{sys.platform}-{os.name}",
        cpu_model="local-cpu",
        cpu_cores_allocated=os.cpu_count() or 4,
        memory_mib=8192,
        python_version=sys.version.split()[0],
        official_lock_hash="local-dev-lock",
        worker_image_digest=None,
    )
