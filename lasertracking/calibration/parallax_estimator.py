"""激光-相机视差距离评估工具（Python 版）

用法::

    python lasertracking/calibration/parallax_estimator.py \\
        --config lasertracking/calibration/parallax.yaml
    python lasertracking/calibration/parallax_estimator.py \\
        --config lasertracking/calibration/parallax.yaml --relative
    python lasertracking/calibration/parallax_estimator.py \\
        --config lasertracking/calibration/parallax.yaml --csv outputs/parallax.csv

数学模型：

    du(Z) = fx * bx / Z
    dv(Z) = fy * by / Z

bx/by 为激光出光点相对相机光心的基线（米），约定
    bx > 0 表示激光在相机光心右侧
    by > 0 表示激光在相机光心下方

若设置了 z_ref，则输出相对偏置 du_res(Z) = du(Z) - du(z_ref)，
即"在 z_ref 处做过 boresight 后，其它距离下残留的像素偏置"。

模块同时作为库使用：``parallax_offset(fx, fy, bx, by, z, z_ref=None)``
返回一个 ``(du, dv)``，供 boresight_calibrator 叠加 overlay。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import yaml


DEFAULT_EPS_LIST: Tuple[float, ...] = (2.0, 3.0, 5.0)


@dataclass
class ParallaxConfig:
    fx: float = 0.0
    fy: float = 0.0
    bx: float = 0.0
    by: float = 0.0
    z_min: float = 1.0
    z_max: float = 50.0
    z_ref: float = 0.0  # 0 或负值视为无穷远
    step: float = 1.0
    eps_list: Tuple[float, ...] = DEFAULT_EPS_LIST

    def validate(self) -> Optional[str]:
        if self.fx <= 0.0 or self.fy <= 0.0:
            return "fx/fy 必须为正"
        if self.z_min <= 0.0 or self.z_max <= self.z_min:
            return "需要 0 < z_min < z_max"
        if self.step <= 0.0:
            return "step 必须为正"
        return None


@dataclass
class ParallaxSample:
    z: float
    du: float
    dv: float


def parallax_offset(fx: float, fy: float, bx: float, by: float,
                    z: float, z_ref: Optional[float] = None) -> Tuple[float, float]:
    """返回距离 z (m) 处激光光斑相对"瞄准点"的像素偏置 (du, dv)。

    若给定 z_ref>0，返回"相对偏置"：即以 z_ref 处为 0 参考。
    """
    if z <= 1e-9:
        return 0.0, 0.0
    du = fx * bx / z
    dv = fy * by / z
    if z_ref is not None and z_ref > 0.0:
        du -= fx * bx / z_ref
        dv -= fy * by / z_ref
    return du, dv


def _load_eps(node) -> Tuple[float, ...]:
    if node is None:
        return DEFAULT_EPS_LIST
    if isinstance(node, (list, tuple)):
        values: List[float] = []
        for v in node:
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                continue
        return tuple(values) if values else DEFAULT_EPS_LIST
    return DEFAULT_EPS_LIST


def _coerce_float(data, key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _normalize_config_data(data: dict) -> dict:
    calibration = data.get("calibration")
    if isinstance(calibration, dict):
        baseline = calibration.get("baseline", {}) or {}
        merged = {
            "fx": calibration.get("fx", data.get("fx", 0.0)),
            "fy": calibration.get("fy", data.get("fy", 0.0)),
            "bx": baseline.get("bx", data.get("bx", 0.0)),
            "by": baseline.get("by", data.get("by", 0.0)),
            "z_ref": baseline.get("z_ref", data.get("z_ref", 0.0)),
            "z_min": data.get("z_min", 1.0),
            "z_max": data.get("z_max", 50.0),
            "step": data.get("step", 1.0),
            "eps_list": data.get("eps_list"),
        }
        return merged
    return data


def load_config(path: str) -> ParallaxConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = _normalize_config_data(yaml.safe_load(f) or {})
    cfg = ParallaxConfig(
        fx=_coerce_float(data, "fx", 0.0),
        fy=_coerce_float(data, "fy", 0.0),
        bx=_coerce_float(data, "bx", 0.0),
        by=_coerce_float(data, "by", 0.0),
        z_min=_coerce_float(data, "z_min", 1.0),
        z_max=_coerce_float(data, "z_max", 50.0),
        z_ref=_coerce_float(data, "z_ref", 0.0),
        step=_coerce_float(data, "step", 1.0),
        eps_list=_load_eps(data.get("eps_list")),
    )
    err = cfg.validate()
    if err is not None:
        raise ValueError(f"parallax 配置不合法: {err}")
    return cfg


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    n = int(math.floor((stop - start) / step + 1e-9)) + 1
    for i in range(max(0, n)):
        yield start + i * step


def evaluate_absolute(cfg: ParallaxConfig) -> List[ParallaxSample]:
    return [
        ParallaxSample(z=z,
                       du=cfg.fx * cfg.bx / z,
                       dv=cfg.fy * cfg.by / z)
        for z in _frange(cfg.z_min, cfg.z_max, cfg.step)
        if z > 0.0
    ]


def evaluate_relative(cfg: ParallaxConfig) -> List[ParallaxSample]:
    z_ref = cfg.z_ref if cfg.z_ref > 0.0 else float("inf")
    du_ref = 0.0 if z_ref == float("inf") else cfg.fx * cfg.bx / z_ref
    dv_ref = 0.0 if z_ref == float("inf") else cfg.fy * cfg.by / z_ref
    return [
        ParallaxSample(z=z,
                       du=(cfg.fx * cfg.bx / z) - du_ref,
                       dv=(cfg.fy * cfg.by / z) - dv_ref)
        for z in _frange(cfg.z_min, cfg.z_max, cfg.step)
        if z > 0.0
    ]


def min_distance_for_eps(cfg: ParallaxConfig, eps_px: float) -> float:
    if eps_px <= 0.0:
        return 0.0
    z_u = 0.0 if abs(cfg.bx) < 1e-9 else cfg.fx * abs(cfg.bx) / eps_px
    z_v = 0.0 if abs(cfg.by) < 1e-9 else cfg.fy * abs(cfg.by) / eps_px
    return max(z_u, z_v)


def relative_range_for_eps(cfg: ParallaxConfig, eps_px: float) -> Tuple[float, float]:
    """给定 eps，返回相对偏置 <= eps 的 [z_min, z_max] 闭区间（米）。

    数学推导：
        |du_res(Z)| = |fx*bx|*|1/Z - 1/z_ref| <= eps
    等价于 1/Z 落在 [1/z_ref - eps/(fx|bx|), 1/z_ref + eps/(fx|bx|)]。
    """
    if eps_px <= 0.0:
        return 0.0, 0.0
    inf = float("inf")
    inv_ref = 0.0 if cfg.z_ref <= 0.0 else 1.0 / cfg.z_ref
    delta_u = inf if abs(cfg.bx) < 1e-9 else eps_px / (cfg.fx * abs(cfg.bx))
    delta_v = inf if abs(cfg.by) < 1e-9 else eps_px / (cfg.fy * abs(cfg.by))
    delta = min(delta_u, delta_v)
    inv_min = max(0.0, inv_ref - delta)
    inv_max = inv_ref + delta
    z_min = 1.0 / inv_max if inv_max > 1e-12 else 0.0
    z_max = 1.0 / inv_min if inv_min > 1e-12 else inf
    return z_min, z_max


def save_csv(path: str, samples: Sequence[ParallaxSample]) -> bool:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    except OSError:
        pass
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("z,du,dv\n")
            for s in samples:
                f.write(f"{s.z:.6f},{s.du:.6f},{s.dv:.6f}\n")
        return True
    except OSError as exc:
        print(f"[parallax] 写 CSV 失败 {path}: {exc}", file=sys.stderr)
        return False


def _fmt_z(z: float) -> str:
    if z == float("inf"):
        return "inf"
    return f"{z:.3f}"


def print_report(cfg: ParallaxConfig, relative: bool) -> List[ParallaxSample]:
    print(f"fx={cfg.fx:.3f} fy={cfg.fy:.3f} bx={cfg.bx:.4f} by={cfg.by:.4f}")
    print(f"z_range=[{cfg.z_min:.2f}, {cfg.z_max:.2f}] step={cfg.step:.2f}")
    print(f"eps_list: {', '.join(f'{e:g}' for e in cfg.eps_list)}")

    if not relative:
        for eps in cfg.eps_list:
            z_min = min_distance_for_eps(cfg, eps)
            print(f"abs eps={eps:g}px -> z >= {z_min:.3f} m")
        samples = evaluate_absolute(cfg)
    else:
        if cfg.z_ref <= 0.0:
            print("warn: relative 模式需要 z_ref>0，否则退化为绝对偏置")
        for eps in cfg.eps_list:
            z_lo, z_hi = relative_range_for_eps(cfg, eps)
            print(f"rel eps={eps:g}px -> z in [{_fmt_z(z_lo)}, {_fmt_z(z_hi)}] m")
        samples = evaluate_relative(cfg)

    if samples:
        first = samples[0]
        last = samples[-1]
        print(f"Samples: {len(samples)}")
        print(f"z={first.z:g} du={first.du:.3f} dv={first.dv:.3f}")
        if last is not first:
            print(f"z={last.z:g} du={last.du:.3f} dv={last.dv:.3f}")
    return samples


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Laser-camera parallax estimator")
    parser.add_argument("--config", required=True, help="parallax.yaml 路径")
    parser.add_argument("--csv", default=None, help="导出 z/du/dv 采样表 CSV 路径")
    parser.add_argument("--relative", action="store_true",
                        help="使用相对偏置（需要 z_ref>0）")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"加载配置失败: {exc}", file=sys.stderr)
        return 1

    samples = print_report(cfg, relative=args.relative)
    if args.csv:
        if save_csv(args.csv, samples):
            print(f"Saved CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
