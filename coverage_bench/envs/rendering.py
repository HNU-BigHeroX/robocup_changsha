"""回合画面的纯 numpy 渲染：无第三方依赖，直接绘制到 RGB 数组。"""
import numpy as np

from coverage_bench.config import TaskConfig
from coverage_bench.envs.types import WorldSnapshot


def render_rgb_frame(world: WorldSnapshot, config: TaskConfig, width: int = 256, height: int = 256) -> np.ndarray:
    """把世界快照画成 RGB 帧：目标为红色圆，机器人为蓝色圆。"""
    frame = np.full((height, width, 3), 245, dtype=np.uint8)
    half_extent = config.public.map_half_extent

    def to_pixel(x: float, y: float) -> tuple[int, int]:
        px = int((x + half_extent) / (2.0 * half_extent) * (width - 1))
        py = int((y + half_extent) / (2.0 * half_extent) * (height - 1))
        return np.clip(px, 0, width - 1), np.clip(py, 0, height - 1)

    scale = width / (2.0 * half_extent)

    # 绘制目标 (红色)
    for j in range(len(world.target_positions)):
        tx, ty = world.target_positions[j]
        tr = float(world.target_radii[j]) * scale
        px, py = to_pixel(tx, ty)
        r_int = max(1, int(tr))
        y_min = max(0, py - r_int)
        y_max = min(height, py + r_int + 1)
        x_min = max(0, px - r_int)
        x_max = min(width, px + r_int + 1)
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (xx - px) ** 2 + (yy - py) ** 2 <= r_int ** 2
        frame[y_min:y_max, x_min:x_max][mask] = [230, 80, 80]

    # 绘制机器人 (蓝色)
    for i in range(len(world.robot_positions)):
        rx, ry = world.robot_positions[i]
        rr = float(world.robot_radii[i]) * scale
        px, py = to_pixel(rx, ry)
        r_int = max(1, int(rr))
        y_min = max(0, py - r_int)
        y_max = min(height, py + r_int + 1)
        x_min = max(0, px - r_int)
        x_max = min(width, px + r_int + 1)
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (xx - px) ** 2 + (yy - py) ** 2 <= r_int ** 2
        frame[y_min:y_max, x_min:x_max][mask] = [50, 100, 220]

    return frame
