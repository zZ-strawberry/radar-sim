"""
从 gimbal_dual_camera_migration_from_sim.yaml 加载仿真/URDF 导出的几何与协议元数据，
供双相机重指向或限位核对使用（与 aerial_tracking_system 解耦，仅依赖 PyYAML + numpy）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class GimbalMigrationBundle:
    """解析后的迁移数据，便于跟踪节点直接取用。"""

    source_path: str
    raw: Dict[str, Any]
    T_short_to_long: np.ndarray  # 4x4, p_short = T @ p_long（齐次）
    translation_short_from_long_m: np.ndarray  # 3，即 T 的平移列（短焦原点看长焦原点）
    baseline_m: float
    pitch_limit_deg: Tuple[float, float]
    yaw_limit_deg: Tuple[float, float]
    roll_left_cap_deg: Optional[float]
    parallax_error_deg_at_distance: Dict[str, float]


def load_gimbal_migration_bundle(
    yaml_path: str,
    roll_left_limit_deg_override: Any = None,
) -> GimbalMigrationBundle:
    if yaml is None:
        raise RuntimeError("需要 PyYAML：请 pip install pyyaml")

    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到几何迁移文件: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"YAML 根节点应为 mapping: {path}")

    t_block = raw.get("T_short_to_long") or {}
    mat = t_block.get("matrix")
    if not mat or len(mat) != 4:
        raise ValueError("T_short_to_long.matrix 缺失或不是 4x4")
    T = np.array(mat, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("T_short_to_long.matrix 必须是 4x4")

    trans = np.array(
        [
            float(T[0, 3]),
            float(T[1, 3]),
            float(T[2, 3]),
        ],
        dtype=np.float64,
    )

    bl = raw.get("baseline_m") or {}
    baseline_m = float(bl.get("optical_centers_euclidean", 0.0))

    ja = raw.get("joint_axes_parent_frame") or {}
    pitch_lim = ((ja.get("pitch_joint") or {}).get("limit_rad")) or {}
    yaw_lim = ((ja.get("yaw_joint") or {}).get("limit_rad")) or {}
    pitch_limit_deg = (
        math.degrees(float(pitch_lim.get("lower", -1.2))),
        math.degrees(float(pitch_lim.get("upper", 1.2))),
    )
    yaw_limit_deg = (
        math.degrees(float(yaw_lim.get("lower", -math.pi))),
        math.degrees(float(yaw_lim.get("upper", math.pi))),
    )

    host = raw.get("host_software_limits") or {}
    cap = host.get("roll_left_limit_deg", None)
    if roll_left_limit_deg_override is not None and roll_left_limit_deg_override != "":
        cap = roll_left_limit_deg_override
    if cap is None or cap == "":
        roll_left_cap: Optional[float] = None
    else:
        roll_left_cap = float(cap)

    pe = (raw.get("parallax_error_deg") or {}).get("at_distance_m") or {}

    return GimbalMigrationBundle(
        source_path=str(path.resolve()),
        raw=raw,
        T_short_to_long=T,
        translation_short_from_long_m=trans,
        baseline_m=baseline_m,
        pitch_limit_deg=pitch_limit_deg,
        yaw_limit_deg=yaw_limit_deg,
        roll_left_cap_deg=roll_left_cap,
        parallax_error_deg_at_distance={str(k): float(v) for k, v in pe.items()},
    )


def print_migration_summary(bundle: GimbalMigrationBundle) -> None:
    print("=" * 60)
    print(" 几何迁移包（仿真 URDF / 串口语义）")
    print("=" * 60)
    print(f" 文件: {bundle.source_path}")
    print(f" 基线(两光学中心): {bundle.baseline_m:.4f} m")
    if bundle.parallax_error_deg_at_distance:
        parts = [f"{d}m≈{a:.3f}°" for d, a in sorted(bundle.parallax_error_deg_at_distance.items(), key=lambda x: float(x[0]))]
        print(f" 视差角(无测距近似): {', '.join(parts)}")
    print(f" 关节限位(度): pitch∈[{bundle.pitch_limit_deg[0]:.2f}, {bundle.pitch_limit_deg[1]:.2f}], "
          f"yaw(上位机 roll)∈[{bundle.yaw_limit_deg[0]:.2f}, {bundle.yaw_limit_deg[1]:.2f}]")
    if bundle.roll_left_cap_deg is not None:
        print(f" 软件向左(roll>0)上限: roll ≤ {bundle.roll_left_cap_deg:.2f}°")
    else:
        print(" 软件向左 roll 上限: 未设置（host_software_limits.roll_left_limit_deg 为 null）")
    t = bundle.translation_short_from_long_m
    print(f" T_short_to_long 平移(m) [短焦系下长焦相对短焦]: [{t[0]:.5f}, {t[1]:.5f}, {t[2]:.5f}]")
    print("=" * 60)


def clip_gimbal_command_deg(
    pitch_deg: float,
    roll_deg: float,
    bundle: Optional[GimbalMigrationBundle],
    apply_joint_limits: bool,
    apply_roll_left_cap: bool,
) -> Tuple[float, float]:
    """按迁移包中的 URDF 限位与可选 roll 左行上限裁剪命令（度）。"""
    p, r = float(pitch_deg), float(roll_deg)
    if bundle is None:
        return p, r
    if apply_joint_limits:
        p = max(bundle.pitch_limit_deg[0], min(bundle.pitch_limit_deg[1], p))
        r = max(bundle.yaw_limit_deg[0], min(bundle.yaw_limit_deg[1], r))
    if apply_roll_left_cap and bundle.roll_left_cap_deg is not None:
        cap = float(bundle.roll_left_cap_deg)
        r = min(r, cap)
    return p, r
