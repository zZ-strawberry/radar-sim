#!/usr/bin/python3
from __future__ import annotations

import math
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as RosTime
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
import yaml

# Ultralytics 在 import 时读取 YOLO_VERBOSE；默认压低其 INFO（需要调试可 export YOLO_VERBOSE=true）。
os.environ.setdefault("YOLO_VERBOSE", "false")

YOLO = None
_ULTRALYTICS_IMPORT_ERROR = ""
try:
    from ultralytics import YOLO as _YOLO

    YOLO = _YOLO
except Exception as exc:  # pragma: no cover
    _ULTRALYTICS_IMPORT_ERROR = str(exc)


@dataclass
class Detection:
    cx: float
    cy: float
    conf: float
    xyxy: Tuple[int, int, int, int]


@dataclass
class TeleDebugOverlay:
    """后台线程绘制长焦调试图所需状态（由主线程在控制逻辑之后快照）。"""

    have_det: bool = False
    det_xyxy: Optional[Tuple[int, int, int, int]] = None
    det_cx: float = 0.0
    det_cy: float = 0.0
    det_conf: float = 0.0
    det_method: str = "none"
    aerial_bbox: Optional[Tuple[int, int, int, int]] = None
    err_x: float = 0.0
    err_y: float = 0.0
    reacq_mode: str = "TRACK"
    tele_lost_count: int = 0
    aim_dx: float = 0.0
    aim_dy: float = 0.0
    boresight_src: str = ""
    yaw: float = 0.0
    pitch: float = 0.0
    reacq_active: bool = False
    parallax_on: bool = False
    parallax_ox: float = 0.0
    parallax_oy: float = 0.0
    parallax_range_str: str = "--"
    parallax_src: str = "none"
    dyn_exp_on: bool = False
    tele_exp_mode: str = "search"
    det_streak: int = 0
    lost_streak: int = 0


@dataclass
class WideDebugOverlay:
    have_det: bool = False
    det_xyxy: Optional[Tuple[int, int, int, int]] = None
    det_cx: float = 0.0
    det_cy: float = 0.0
    det_conf: float = 0.0


class PredictiveKalmanFilter:
    """与实车版本一致的双轴预测卡尔曼（角度域）。"""

    def __init__(
        self,
        prediction_time_ms: float = 80.0,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-1,
        max_inactive_time: float = 2.0,
        max_rate_pitch: Optional[float] = None,
        max_rate_roll: Optional[float] = None,
        settle_pitch_deg: Optional[float] = None,
        settle_roll_deg: Optional[float] = None,
        settle_velocity_damp: float = 0.0,
    ) -> None:
        self.prediction_time = float(prediction_time_ms) / 1000.0
        self.max_inactive_time = float(max_inactive_time)
        self.max_rate_pitch = max_rate_pitch
        self.max_rate_roll = max_rate_roll
        self.settle_pitch_deg = settle_pitch_deg
        self.settle_roll_deg = settle_roll_deg
        self.settle_velocity_damp = float(settle_velocity_damp)

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(process_noise)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(measurement_noise)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)

        self.initialized = False
        self.last_update_time: Optional[float] = None
        self.last_measurement: Optional[np.ndarray] = None

    def _update_transition_matrix(self, dt: float) -> None:
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )

    def reset(self) -> None:
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.last_update_time = None
        self.last_measurement = None

    def update(self, pitch_deg: float, roll_deg: float) -> tuple[float, float]:
        now = time.time()
        if self.last_update_time and (now - self.last_update_time) > self.max_inactive_time:
            self.reset()

        if not self.initialized:
            self.kf.statePost = np.array([[pitch_deg], [roll_deg], [0.0], [0.0]], dtype=np.float32)
            self.initialized = True
            self.last_update_time = now
            self.last_measurement = np.array([pitch_deg, roll_deg], dtype=np.float32)
            return pitch_deg, roll_deg

        dt = max(1e-3, now - (self.last_update_time or now))
        m_pitch = float(pitch_deg)
        m_roll = float(roll_deg)
        if self.last_measurement is not None:
            if self.max_rate_pitch is not None:
                lim = self.max_rate_pitch * dt
                d = m_pitch - float(self.last_measurement[0])
                if abs(d) > lim:
                    m_pitch = float(self.last_measurement[0]) + math.copysign(lim, d)
            if self.max_rate_roll is not None:
                lim = self.max_rate_roll * dt
                d = m_roll - float(self.last_measurement[1])
                if abs(d) > lim:
                    m_roll = float(self.last_measurement[1]) + math.copysign(lim, d)

        self._update_transition_matrix(dt)
        self.kf.predict()
        self.kf.correct(np.array([[m_pitch], [m_roll]], dtype=np.float32))

        # 测量已进入小误差带时衰减速度状态，减轻「框抖动→假角速度→prediction 超前」导致的静止过冲
        if (
            self.settle_velocity_damp > 0.0
            and self.settle_pitch_deg is not None
            and self.settle_roll_deg is not None
            and abs(m_pitch) <= self.settle_pitch_deg
            and abs(m_roll) <= self.settle_roll_deg
        ):
            self.kf.statePost[2, 0] *= self.settle_velocity_damp
            self.kf.statePost[3, 0] *= self.settle_velocity_damp

        pred_pitch = float(self.kf.statePost[0, 0] + self.kf.statePost[2, 0] * self.prediction_time)
        pred_roll = float(self.kf.statePost[1, 0] + self.kf.statePost[3, 0] * self.prediction_time)
        self.last_update_time = now
        self.last_measurement = np.array([m_pitch, m_roll], dtype=np.float32)
        return pred_pitch, pred_roll

    def get_velocity_deg_s(self) -> tuple[float, float]:
        if not self.initialized:
            return 0.0, 0.0
        return float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])

    def zero_velocity(self) -> None:
        if not self.initialized:
            return
        self.kf.statePost[2, 0] = 0.0
        self.kf.statePost[3, 0] = 0.0


class LaserTrackingTracker(Node):
    def __init__(self) -> None:
        super().__init__("lasertracking_tracker")

        self.declare_parameter("model_path", "")
        self.declare_parameter("wide_image_topic", "/fixed_camera/image_raw")
        self.declare_parameter("tele_image_topic", "/gimbal_camera/image_raw")
        # 短焦整链总闸：false 时不订阅短焦、不加载短焦模型、不启短焦检测线程、不创建短焦相关发布器，
        # 物理接上短焦/有图像话题亦完全忽略；并关闭重获中的短焦辅助（与 enable_wide_detection 等为 AND）。
        # 默认 false=仅长焦；需短焦时在 YAML 设 enable_short_focal_pipeline:=true。海康见 hw_sdk start_wide_camera；长焦序号见 hik_tele.yaml device_index。
        self.declare_parameter("enable_short_focal_pipeline", False)
        self.declare_parameter("enable_wide_detection", True)
        self.declare_parameter("upper_cmd_topic", "/lasertracking/upper_cmd")
        self.declare_parameter("runtime_mode", "sim")  # sim | hw
        self.declare_parameter("publish_upper_cmd_in_hw", False)
        self.declare_parameter("hw_comm_link", "serial")  # serial | usb（usb=CDC 虚拟串口，仍用 PySerial）
        self.declare_parameter("hw_usb_serial_port", "")
        self.declare_parameter("hw_serial_port", "/dev/ttyUSB0")
        self.declare_parameter("hw_serial_baudrate", 115200)
        self.declare_parameter("hw_serial_protocol", "raw")
        self.declare_parameter("hw_serial_enable_feedback", False)
        self.declare_parameter("hw_use_feedback_state", False)
        self.declare_parameter("hw_feedback_max_age_s", 0.20)
        self.declare_parameter("debug_image_topic", "/lasertracking/debug_image")
        self.declare_parameter("wide_debug_image_topic", "/lasertracking/wide_debug_image")
        self.declare_parameter("wide_detection_topic", "/lasertracking/wide_detection")
        # 调试图异步发布：在独立线程中 tobytes + publish，减轻 rclpy 主线程单核瓶颈
        self.declare_parameter("enable_async_debug_publish", True)
        self.declare_parameter("async_debug_drop_if_busy", True)
        self.declare_parameter(
            "tele_debug_image_max_width",
            0,
        )  # >0 时发布前缩小长焦调试图，减轻编码与 async 丢帧，不影响检测/跟踪分辨率
        self.declare_parameter(
            "wide_debug_image_max_width",
            0,
        )  # >0 时发布前缩小短焦调试图，逻辑同 tele_debug_image_max_width
        self.declare_parameter("enable_debug_image_publish", True)
        self.declare_parameter("enable_tele_debug_image_publish", True)
        self.declare_parameter("enable_wide_debug_image_publish", True)
        # 长焦原图内录：回调只拷贝原始 payload 入队，Bayer/BGR 解码与 VideoWriter 均在后台线程（减轻对话题回调与主循环的影响）
        self.declare_parameter("enable_tele_raw_recording", True)
        self.declare_parameter(
            "tele_recording_output_dir",
            "",
        )  # 空=自动使用「cwd/recordings/tele_raw」并 mkdir；非空=视为已存在的输出目录，不自动创建，仅校验存在且可写
        self.declare_parameter("tele_recording_fps", 25.0)
        # 默认 MJPG：Motion JPEG + AVI，帧内编码、OpenCV 下通常比 mp4v 更省交互延迟与异常 GOP 依赖；重编码仅在写盘线程
        self.declare_parameter("tele_recording_fourcc", "MJPG")
        # 空=按 fourcc 自动选 .avi / .mp4；也可显式写 ".avi" / ".mkv"（须与本机 OpenCV 后端匹配）
        self.declare_parameter("tele_recording_extension", "")
        self.declare_parameter("tele_recording_queue_max", 48)
        # 每 N 帧长焦图才入队 1 次（仅影响录制，不影响检测）；2=约减半拷贝与写盘负载
        self.declare_parameter("tele_recording_subsample", 1)
        # 每次节点运行在长焦落盘根目录下新建 {prefix}_时间戳 子文件夹，按帧保存长焦原图（后台解码+imwrite，不改检测/控制逻辑）
        self.declare_parameter("enable_tele_game_folder_images", False)
        self.declare_parameter("tele_game_folder_prefix", "game")
        self.declare_parameter("tele_game_image_queue_max", 32)
        self.declare_parameter("tele_game_image_subsample", 1)
        self.declare_parameter("tele_game_image_format", "jpg")  # jpg | png
        self.declare_parameter("tele_game_jpeg_quality", 92)
        # 与 hik_wide.yaml / hik_tele.yaml 对齐的台账（节点不直接驱动相机；长焦设备枚举序号见 hik_tele.yaml device_index）
        self.declare_parameter("hw_hik_wide_exposure_time", 39000.0)
        self.declare_parameter("hw_hik_wide_gain", 16.0)
        self.declare_parameter("hw_hik_tele_exposure_time", 35000.0)
        self.declare_parameter("hw_hik_tele_gain", 20.0)
        self.declare_parameter("loop_hz", 20.0)
        self.declare_parameter("tele_detect_hz", 20.0)
        self.declare_parameter("wide_detect_hz", 5.0)
        self.declare_parameter("wide_detect_hz_idle", 1.0)
        self.declare_parameter("wide_detect_hz_reacq", 11.0)
        self.declare_parameter("use_tensorrt", False)
        self.declare_parameter("trt_engine_path", "")
        self.declare_parameter("tele_model_path", "")
        self.declare_parameter("tele_trt_engine_path", "")
        self.declare_parameter("tele_aerial_model_path", "lasertracking/model/aerial.pt")
        self.declare_parameter("tele_aerial_trt_engine_path", "")
        self.declare_parameter("tele_aerial_min_conf", 0.10)
        self.declare_parameter("tele_aerial_infer_device", "cuda:0")
        self.declare_parameter("tele_aerial_infer_imgsz", 640)
        self.declare_parameter("tele_aerial_infer_half", True)
        self.declare_parameter("tele_infer_device", "cuda:0")
        self.declare_parameter("tele_infer_imgsz", 640)
        self.declare_parameter("tele_infer_half", True)
        self.declare_parameter("tele_min_conf", 0.25)
        self.declare_parameter("tele_direct_conf_gate", 0.70)
        self.declare_parameter("wide_model_path", "")
        self.declare_parameter("wide_trt_engine_path", "")
        self.declare_parameter("wide_infer_device", "cuda:0")
        self.declare_parameter("wide_infer_imgsz", 320)
        self.declare_parameter("wide_infer_half", True)
        self.declare_parameter("wide_min_conf", 0.25)
        self.declare_parameter("wide_detect_right_half_only", True)
        self.declare_parameter("kp_yaw", 0.9)
        self.declare_parameter("kp_pitch", 0.9)
        self.declare_parameter("command_smooth_yaw", 0.70)
        self.declare_parameter("command_smooth_pitch", 0.65)
        self.declare_parameter("deadzone_yaw_deg", 0.20)
        self.declare_parameter("deadzone_pitch_deg", 0.20)
        self.declare_parameter("pitch_deadzone_exit_ratio", 1.25)
        self.declare_parameter("enable_velocity_feedforward", False)
        self.declare_parameter("velocity_decay_multiple", 5.0)
        self.declare_parameter("enable_kalman_filter", True)
        self.declare_parameter("kalman_prediction_time_ms", 60.0)
        self.declare_parameter("kalman_process_noise", 1e-3)
        self.declare_parameter("kalman_measurement_noise", 1e-1)
        self.declare_parameter("kalman_max_rate_pitch_deg_s", 180.0)
        self.declare_parameter("kalman_max_rate_roll_deg_s", 180.0)
        self.declare_parameter("yaw_error_sign", -1.0)
        self.declare_parameter("pitch_error_sign", 1.0)
        self.declare_parameter("max_step_rad", 0.04)
        self.declare_parameter("yaw_limit_rad", 3.0)
        self.declare_parameter("yaw_left_limit_rad", 3.0)
        self.declare_parameter("pitch_limit_rad", 1.1)
        self.declare_parameter("tele_hfov_deg", 8.3)
        self.declare_parameter("tele_vfov_deg", 6.2)
        self.declare_parameter("wide_hfov_deg", 56.0)
        self.declare_parameter("wide_vfov_deg", 42.0)
        self.declare_parameter("enable_reacq_assist", True)
        self.declare_parameter("enable_reacq_scan", True)
        self.declare_parameter("enable_reacq_wide_assist", True)
        self.declare_parameter("reacq_require_first_lock", True)
        self.declare_parameter("reacq_enter_lost_frames", 3)
        self.declare_parameter("reacq_wide_enter_lost_frames", 7)
        self.declare_parameter("reacq_confirm_frames", 2)
        self.declare_parameter("reacq_use_wide_detection", True)
        self.declare_parameter("reacq_max_step_rad", 0.03)
        self.declare_parameter("reacq_scan_fallback", True)
        # 丢失扫描螺旋：参数为角度(度)，加载时换算为弧度参与内部计算
        self.declare_parameter("reacq_scan_spiral_spacing_deg", math.degrees(0.015))
        # 内圈：前几整圈用较小每圈半径增量(度)，避免一上来圈太大漏扫中心；0 圈内则与外层相同
        self.declare_parameter("reacq_scan_spiral_spacing_start_deg", 3.0)
        self.declare_parameter("reacq_scan_spiral_inner_turns", 2.0)
        self.declare_parameter("reacq_scan_spiral_speed_deg_s", math.degrees(0.20))
        self.declare_parameter("reacq_scan_spiral_r_max_deg", math.degrees(0.18))
        self.declare_parameter("reacq_scan_spiral_yaw_scale", 1.0)
        self.declare_parameter("reacq_scan_spiral_pitch_scale", 1.0)
        self.declare_parameter("reacq_scan_spiral_theta_max_deg", math.degrees(7.85))
        self.declare_parameter("reacq_wide_lost_tolerance_frames", 4)
        self.declare_parameter("reacq_scan_to_wide_frames", 20)
        self.declare_parameter("reacq_wide_phase_frames", 18)
        self.declare_parameter("reacq_wide_map_m00", 1.0)
        self.declare_parameter("reacq_wide_map_m01", 0.0)
        self.declare_parameter("reacq_wide_map_m10", 0.0)
        self.declare_parameter("reacq_wide_map_m11", 1.0)
        self.declare_parameter("reacq_wide_ref_yaw_rad", 0.0)
        self.declare_parameter("reacq_wide_ref_pitch_rad", 0.0)
        # 丢失螺旋扫描预设原点：相对上电初始的绝对 yaw/pitch，YAML 为度
        self.declare_parameter("reacq_scan_preset_yaw_deg", 0.0)
        self.declare_parameter("reacq_scan_preset_pitch_deg", 0.0)
        self.declare_parameter("reacq_scan_use_preset_without_anchor", True)
        self.declare_parameter("reacq_scan_use_preset_after_total_lost_frames", 0)
        # >0 时按「无长焦检测的控制周期」换算为秒：阈值=round(秒×loop_hz)，并覆盖上一项
        self.declare_parameter("reacq_scan_use_preset_after_lost_seconds", 0.0)
        self.declare_parameter("pixel_offset_x", 0.0)
        self.declare_parameter("pixel_offset_y", 0.0)
        self.declare_parameter("boresight_file", "")
        self.declare_parameter("calibration_fx", 0.0)
        self.declare_parameter("calibration_fy", 0.0)
        self.declare_parameter("baseline_bx", 0.0)
        self.declare_parameter("baseline_by", 0.0)
        self.declare_parameter("baseline_z_ref", 0.0)
        self.declare_parameter("parallax_enabled", False)
        self.declare_parameter("parallax_source", "fixed")
        self.declare_parameter("parallax_fixed_distance_m", 20.0)
        self.declare_parameter("parallax_target_width_m", 0.0)
        self.declare_parameter("parallax_target_height_m", 0.0)
        self.declare_parameter("parallax_min_distance_m", 1.0)
        self.declare_parameter("parallax_max_distance_m", 100.0)
        self.declare_parameter("parallax_distance_alpha", 0.35)
        self.declare_parameter("parallax_hold_last_distance", True)
        self.declare_parameter("parallax_min_bbox_px", 8.0)
        # 图像预处理曝光增强：长焦搜索高/锁定低，短焦固定偏高
        self.declare_parameter("enable_dynamic_tele_exposure", False)
        self.declare_parameter("tele_exposure_search_gain", 1.0)
        self.declare_parameter("tele_exposure_track_gain", 1.0)
        self.declare_parameter("tele_exposure_search_offset", 0.0)
        self.declare_parameter("tele_exposure_track_offset", 0.0)
        self.declare_parameter("tele_exposure_lock_frames", 3)
        self.declare_parameter("tele_exposure_lost_frames", 5)
        self.declare_parameter("tele_exposure_min_switch_interval_s", 0.6)
        self.declare_parameter("wide_exposure_gain", 1.0)
        self.declare_parameter("wide_exposure_offset", 0.0)

        self._enable_short_focal_pipeline = bool(
            self.get_parameter("enable_short_focal_pipeline").value
        )
        self._enable_wide_detection = bool(
            self.get_parameter("enable_wide_detection").value
        ) and self._enable_short_focal_pipeline

        base_pt = str(self.get_parameter("model_path").value).strip()
        if not base_pt:
            raise RuntimeError("参数 model_path 不能为空")

        tele_pt = str(self.get_parameter("tele_model_path").value).strip() or base_pt
        tele_aerial_pt = str(self.get_parameter("tele_aerial_model_path").value).strip()
        wide_pt = str(self.get_parameter("wide_model_path").value).strip() or base_pt
        tele_trt_ov = str(self.get_parameter("tele_trt_engine_path").value).strip()
        tele_aerial_trt_ov = str(self.get_parameter("tele_aerial_trt_engine_path").value).strip()
        wide_trt_ov = str(self.get_parameter("wide_trt_engine_path").value).strip()

        tele_w = self._resolve_weight_for_branch(tele_pt, tele_trt_ov)
        tele_aerial_w = self._resolve_weight_for_branch(tele_aerial_pt, tele_aerial_trt_ov) if tele_aerial_pt else None
        wide_w = self._resolve_weight_for_branch(wide_pt, wide_trt_ov)

        self._model_tele: Optional[object] = None
        self._model_wide: Optional[object] = None
        self._model_tele_aerial: Optional[object] = None
        self._infer_lock = threading.Lock()
        self._detector_ready = False
        self._detector_wide_ready = False
        self._last_missing_dep_warn_ns = 0
        self._last_runtime_error_ns = 0
        self._last_tele_dbg_decode_warn_ns = 0
        self._last_image_decode_warn_enc: Optional[str] = None

        self._predict_kwargs_tele = {
            "verbose": False,
            "device": str(self.get_parameter("tele_infer_device").value),
            "imgsz": int(self.get_parameter("tele_infer_imgsz").value),
            "half": bool(self.get_parameter("tele_infer_half").value),
        }
        self._predict_kwargs_tele_aerial = {
            "verbose": False,
            "device": str(self.get_parameter("tele_aerial_infer_device").value),
            "imgsz": int(self.get_parameter("tele_aerial_infer_imgsz").value),
            "half": bool(self.get_parameter("tele_aerial_infer_half").value),
        }
        self._predict_kwargs_wide = {
            "verbose": False,
            "device": str(self.get_parameter("wide_infer_device").value),
            "imgsz": int(self.get_parameter("wide_infer_imgsz").value),
            "half": bool(self.get_parameter("wide_infer_half").value),
        }
        self._min_conf_tele = float(self.get_parameter("tele_min_conf").value)
        self._tele_direct_conf_gate = float(self.get_parameter("tele_direct_conf_gate").value)
        self._min_conf_tele_aerial = float(self.get_parameter("tele_aerial_min_conf").value)
        self._min_conf_wide = float(self.get_parameter("wide_min_conf").value)
        self._wide_detect_right_half_only = bool(
            self.get_parameter("wide_detect_right_half_only").value
        )
        self._wide_detect_hz_base = max(0.2, float(self.get_parameter("wide_detect_hz").value))
        self._wide_detect_hz_idle = max(
            0.2, float(self.get_parameter("wide_detect_hz_idle").value)
        )
        reacq_hz_raw = float(self.get_parameter("wide_detect_hz_reacq").value)
        self._wide_detect_hz_reacq = (
            self._wide_detect_hz_base if reacq_hz_raw <= 0.0 else max(0.2, reacq_hz_raw)
        )
        self._wide_detect_hz_current = self._wide_detect_hz_idle
        self._wide_hz_mode = "idle"

        use_tensorrt = bool(self.get_parameter("use_tensorrt").value)
        if YOLO is None:
            hint = (
                "请执行: /usr/bin/python3 -m pip install ultralytics；"
                "若已安装仍失败，多为缺少 PyTorch："
                "/usr/bin/python3 -m pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cpu"
            )
            self.get_logger().error(
                "无法加载 YOLO 检测后端（ultralytics 导入失败），lasertracking_tracker 将保持空转。"
                f" 原因: {_ULTRALYTICS_IMPORT_ERROR or '(unknown)'}。{hint}"
            )
        elif tele_w:
            try:
                self._model_tele = YOLO(tele_w)
                self._detector_ready = True
                self.get_logger().info(
                    f"长焦检测模型: {tele_w} device={self._predict_kwargs_tele['device']} "
                    f"imgsz={self._predict_kwargs_tele['imgsz']} half={self._predict_kwargs_tele['half']}"
                )
                if use_tensorrt and str(tele_w).endswith(".engine"):
                    self.get_logger().info("长焦 TensorRT engine 已加载")
            except Exception as exc:
                self.get_logger().error(f"长焦模型加载失败: {exc}")

        if YOLO is not None and tele_aerial_w:
            try:
                self._model_tele_aerial = YOLO(tele_aerial_w)
                self.get_logger().info(
                    f"长焦无人机模型: {tele_aerial_w} "
                    f"device={self._predict_kwargs_tele_aerial['device']} "
                    f"imgsz={self._predict_kwargs_tele_aerial['imgsz']} "
                    f"half={self._predict_kwargs_tele_aerial['half']}"
                )
            except Exception as exc:
                self._model_tele_aerial = None
                self.get_logger().error(f"长焦无人机模型加载失败: {exc}")

        if self._enable_wide_detection and wide_w and YOLO is not None:
            if wide_w == tele_w and self._model_tele is not None:
                self._model_wide = self._model_tele
                self._detector_wide_ready = True
                self.get_logger().info("短焦与长焦共用同一权重文件，推理参数仍按 wide_* 独立")
            elif wide_w != tele_w:
                try:
                    self._model_wide = YOLO(wide_w)
                    self._detector_wide_ready = True
                    self.get_logger().info(
                        f"短焦检测模型: {wide_w} device={self._predict_kwargs_wide['device']} "
                        f"imgsz={self._predict_kwargs_wide['imgsz']} half={self._predict_kwargs_wide['half']}"
                    )
                except Exception as exc:
                    self.get_logger().error(f"短焦模型加载失败: {exc}")
        elif self._enable_wide_detection and not wide_w:
            self.get_logger().error("短焦检测已启用但短焦权重路径无法解析，请检查 YAML")

        if (
            self._enable_wide_detection
            and self._detector_wide_ready
            and self._model_wide is self._model_tele
            and self._model_tele is not None
            and use_tensorrt
            and tele_w
            and str(tele_w).endswith(".engine")
            and int(self._predict_kwargs_tele["imgsz"]) != int(self._predict_kwargs_wide["imgsz"])
        ):
            self.get_logger().warn(
                "短焦与长焦共用同一 TensorRT engine，但 tele_infer_imgsz ≠ wide_infer_imgsz，"
                "推理可能失败；请为短焦设置 wide_trt_engine_path 或改用 .pt。"
            )

        self._log_torch_cuda_vs_infer_devices()

        self._wide_latest: Optional[Image] = None
        self._tele_latest: Optional[Image] = None
        self._det_lock = threading.Lock()
        self._tele_det_cached: Optional[Detection] = None
        self._tele_size_cached: Optional[Tuple[int, int]] = None
        self._tele_det_method_cached: str = "none"
        self._tele_aerial_bbox_cached: Optional[Tuple[int, int, int, int]] = None
        self._wide_det_cached: Optional[Detection] = None
        self._wide_size_cached: Optional[Tuple[int, int]] = None
        self._running = True
        self._tele_worker: Optional[threading.Thread] = None
        self._wide_worker: Optional[threading.Thread] = None
        self._tele_record_queue: Optional[queue.Queue] = None
        self._tele_record_thread: Optional[threading.Thread] = None
        self._tele_record_running = False
        self._tele_record_file: Optional[Path] = None
        self._tele_record_fps: float = 25.0
        self._tele_record_fourcc: str = "MJPG"
        self._tele_record_queue_full_warned = False
        self._tele_record_decode_fail = 0
        self._tele_record_logged_first_frame = False
        self._tele_record_subsample = 1
        self._tele_record_subsample_counter = 0
        self._tele_game_queue: Optional[queue.Queue] = None
        self._tele_game_thread: Optional[threading.Thread] = None
        self._tele_game_running = False
        self._tele_game_dir: Optional[Path] = None
        self._tele_game_subsample = 1
        self._tele_game_subsample_counter = 0
        self._tele_game_queue_full_warned = False
        self._tele_game_decode_fail = 0
        self._tele_game_logged_first = False
        self._tele_game_ext = ".jpg"
        self._tele_game_jpeg_quality = 92
        self._serial = None
        self._runtime_mode = str(self.get_parameter("runtime_mode").value).strip().lower()
        self._publish_upper_cmd_in_hw = bool(self.get_parameter("publish_upper_cmd_in_hw").value)
        self._hw_serial_enable_feedback = bool(
            self.get_parameter("hw_serial_enable_feedback").value
        )
        self._hw_use_feedback_state = bool(self.get_parameter("hw_use_feedback_state").value)
        self._hw_feedback_max_age_s = max(
            0.01, float(self.get_parameter("hw_feedback_max_age_s").value)
        )
        if self._runtime_mode not in ("sim", "hw"):
            self.get_logger().warn(f"未知 runtime_mode={self._runtime_mode}，回退为 sim")
            self._runtime_mode = "sim"
        if self._runtime_mode == "hw":
            self._init_hw_serial()

        self._yaw = 0.0
        self._pitch = 0.0
        self._tele_hfov = math.radians(float(self.get_parameter("tele_hfov_deg").value))
        self._tele_vfov = math.radians(float(self.get_parameter("tele_vfov_deg").value))
        self._wide_hfov = math.radians(float(self.get_parameter("wide_hfov_deg").value))
        self._wide_vfov = math.radians(float(self.get_parameter("wide_vfov_deg").value))
        self._kp_yaw = float(self.get_parameter("kp_yaw").value)
        self._kp_pitch = float(self.get_parameter("kp_pitch").value)
        self._smooth_yaw = float(self.get_parameter("command_smooth_yaw").value)
        self._smooth_pitch = float(self.get_parameter("command_smooth_pitch").value)
        self._dz_yaw = float(self.get_parameter("deadzone_yaw_deg").value)
        self._dz_pitch = float(self.get_parameter("deadzone_pitch_deg").value)
        self._dz_pitch_exit_ratio = float(self.get_parameter("pitch_deadzone_exit_ratio").value)
        self._pitch_deadzone_latched = False
        self._enable_velocity_ff = bool(self.get_parameter("enable_velocity_feedforward").value)
        self._velocity_decay_multiple = float(self.get_parameter("velocity_decay_multiple").value)
        self._kf = PredictiveKalmanFilter(
            prediction_time_ms=float(self.get_parameter("kalman_prediction_time_ms").value),
            process_noise=float(self.get_parameter("kalman_process_noise").value),
            measurement_noise=float(self.get_parameter("kalman_measurement_noise").value),
            max_rate_pitch=float(self.get_parameter("kalman_max_rate_pitch_deg_s").value),
            max_rate_roll=float(self.get_parameter("kalman_max_rate_roll_deg_s").value),
            settle_pitch_deg=2.8 * self._dz_pitch,
            settle_roll_deg=2.8 * self._dz_yaw,
            settle_velocity_damp=0.18,
        )
        self._enable_kalman = bool(self.get_parameter("enable_kalman_filter").value)
        self._enable_reacq_assist = bool(self.get_parameter("enable_reacq_assist").value)
        self._enable_reacq_scan = bool(self.get_parameter("enable_reacq_scan").value)
        self._enable_reacq_wide_assist = bool(
            self.get_parameter("enable_reacq_wide_assist").value
        ) and self._enable_short_focal_pipeline
        self._reacq_require_first_lock = bool(
            self.get_parameter("reacq_require_first_lock").value
        )
        self._reacq_enter_lost_frames = max(
            1, int(self.get_parameter("reacq_enter_lost_frames").value)
        )
        self._reacq_wide_enter_lost_frames = max(
            self._reacq_enter_lost_frames,
            int(self.get_parameter("reacq_wide_enter_lost_frames").value),
        )
        self._reacq_confirm_frames = max(1, int(self.get_parameter("reacq_confirm_frames").value))
        self._reacq_use_wide = bool(
            self.get_parameter("reacq_use_wide_detection").value
        ) and self._enable_short_focal_pipeline
        self._reacq_max_step = max(0.001, float(self.get_parameter("reacq_max_step_rad").value))
        self._reacq_scan_fallback = bool(self.get_parameter("reacq_scan_fallback").value)
        self._reacq_scan_spiral_spacing = max(
            1e-4, math.radians(float(self.get_parameter("reacq_scan_spiral_spacing_deg").value))
        )
        spacing_start_raw = max(
            1e-5, math.radians(float(self.get_parameter("reacq_scan_spiral_spacing_start_deg").value))
        )
        # 内圈步进不大于外圈，避免配置反了
        self._reacq_scan_spiral_spacing_start = min(spacing_start_raw, self._reacq_scan_spiral_spacing)
        inner_turns = max(0.0, float(self.get_parameter("reacq_scan_spiral_inner_turns").value))
        self._reacq_scan_spiral_theta1 = inner_turns * 2.0 * math.pi
        self._reacq_scan_spiral_a_inner = self._reacq_scan_spiral_spacing_start / (2.0 * math.pi)
        self._reacq_scan_spiral_a_outer = self._reacq_scan_spiral_spacing / (2.0 * math.pi)
        self._reacq_scan_spiral_speed = max(
            0.01, math.radians(float(self.get_parameter("reacq_scan_spiral_speed_deg_s").value))
        )
        self._reacq_scan_spiral_r_max = max(
            0.005, math.radians(float(self.get_parameter("reacq_scan_spiral_r_max_deg").value))
        )
        self._reacq_scan_spiral_yaw_scale = max(
            0.2, float(self.get_parameter("reacq_scan_spiral_yaw_scale").value)
        )
        self._reacq_scan_spiral_pitch_scale = max(
            0.2, float(self.get_parameter("reacq_scan_spiral_pitch_scale").value)
        )
        self._reacq_scan_spiral_theta_max = max(
            math.pi, math.radians(float(self.get_parameter("reacq_scan_spiral_theta_max_deg").value))
        )
        self._reacq_wide_lost_tolerance = max(
            0, int(self.get_parameter("reacq_wide_lost_tolerance_frames").value)
        )
        self._reacq_scan_to_wide_frames = max(
            1, int(self.get_parameter("reacq_scan_to_wide_frames").value)
        )
        self._reacq_wide_phase_frames = max(
            1, int(self.get_parameter("reacq_wide_phase_frames").value)
        )
        self._reacq_wide_map_m00 = float(self.get_parameter("reacq_wide_map_m00").value)
        self._reacq_wide_map_m01 = float(self.get_parameter("reacq_wide_map_m01").value)
        self._reacq_wide_map_m10 = float(self.get_parameter("reacq_wide_map_m10").value)
        self._reacq_wide_map_m11 = float(self.get_parameter("reacq_wide_map_m11").value)
        self._reacq_wide_ref_yaw = float(self.get_parameter("reacq_wide_ref_yaw_rad").value)
        self._reacq_wide_ref_pitch = float(self.get_parameter("reacq_wide_ref_pitch_rad").value)
        self._reacq_scan_preset_yaw = math.radians(float(self.get_parameter("reacq_scan_preset_yaw_deg").value))
        self._reacq_scan_preset_pitch = math.radians(
            float(self.get_parameter("reacq_scan_preset_pitch_deg").value)
        )
        self._reacq_scan_use_preset_without_anchor = bool(
            self.get_parameter("reacq_scan_use_preset_without_anchor").value
        )
        lost_sec = max(0.0, float(self.get_parameter("reacq_scan_use_preset_after_lost_seconds").value))
        lost_frames = max(0, int(self.get_parameter("reacq_scan_use_preset_after_total_lost_frames").value))
        hz_for_thr = max(1.0, float(self.get_parameter("loop_hz").value))
        if lost_sec > 0.0:
            self._reacq_scan_use_preset_after_total_lost_frames = max(
                1, int(round(lost_sec * hz_for_thr))
            )
        else:
            self._reacq_scan_use_preset_after_total_lost_frames = lost_frames
        if lost_sec > 0.0:
            self.get_logger().info(
                "重获扫描迁预设原点: "
                f"reacq_scan_use_preset_after_lost_seconds={lost_sec:g}s × loop_hz={hz_for_thr:.3g} "
                f"→ 无长焦检测 {self._reacq_scan_use_preset_after_total_lost_frames} 周期后迁原点并重卷螺旋"
            )
        self._reacq_active = False
        self._tele_lost_count = 0
        self._reacq_seen_count = 0
        self._reacq_scan_theta = 0.0
        self._reacq_scan_center_yaw = 0.0
        self._reacq_scan_center_pitch = 0.0
        self._tele_loss_anchor_valid = False
        self._tele_loss_anchor_yaw = 0.0
        self._tele_loss_anchor_pitch = 0.0
        self._reacq_last_wide_err: Optional[Tuple[float, float]] = None
        self._reacq_wide_miss_count = 0
        self._reacq_scan_frames = 0
        self._reacq_wide_frames = 0
        self._reacq_phase = "scan"
        self._last_timer_ts = time.time()
        self._debug_overlay_err_x = 0.0
        self._debug_overlay_err_y = 0.0
        self._debug_overlay_reacq_mode = "TRACK"
        self._tele_has_lock_once = False
        self._base_aim_pixel_offset = (
            float(self.get_parameter("pixel_offset_x").value),
            float(self.get_parameter("pixel_offset_y").value),
        )
        self._aim_pixel_offset = self._base_aim_pixel_offset
        self._boresight_source = "legacy"

        self._calibration_fx = float(self.get_parameter("calibration_fx").value)
        self._calibration_fy = float(self.get_parameter("calibration_fy").value)
        self._parallax_enabled = bool(self.get_parameter("parallax_enabled").value)
        self._parallax_source = str(self.get_parameter("parallax_source").value).strip().lower()
        self._parallax_fixed_distance_m = float(self.get_parameter("parallax_fixed_distance_m").value)
        self._parallax_target_width_m = float(self.get_parameter("parallax_target_width_m").value)
        self._parallax_target_height_m = float(self.get_parameter("parallax_target_height_m").value)
        self._parallax_min_distance_m = float(self.get_parameter("parallax_min_distance_m").value)
        self._parallax_max_distance_m = float(self.get_parameter("parallax_max_distance_m").value)
        self._parallax_distance_alpha = float(self.get_parameter("parallax_distance_alpha").value)
        self._parallax_hold_last_distance = bool(self.get_parameter("parallax_hold_last_distance").value)
        self._parallax_min_bbox_px = float(self.get_parameter("parallax_min_bbox_px").value)
        self._parallax_baseline_bx = float(self.get_parameter("baseline_bx").value)
        self._parallax_baseline_by = float(self.get_parameter("baseline_by").value)
        self._parallax_z_ref = float(self.get_parameter("baseline_z_ref").value)
        self._parallax_last_distance_m: Optional[float] = None
        self._parallax_last_distance_source: str = "none"
        self._parallax_last_offset = (0.0, 0.0)
        self._enable_dynamic_tele_exposure = bool(
            self.get_parameter("enable_dynamic_tele_exposure").value
        )
        self._tele_exposure_mode = "search"
        self._tele_detect_streak = 0
        self._tele_lost_streak = 0
        self._last_exposure_switch_ts = 0.0
        self._tele_exposure_search_gain = float(
            self.get_parameter("tele_exposure_search_gain").value
        )
        self._tele_exposure_track_gain = float(
            self.get_parameter("tele_exposure_track_gain").value
        )
        self._tele_exposure_search_offset = float(
            self.get_parameter("tele_exposure_search_offset").value
        )
        self._tele_exposure_track_offset = float(
            self.get_parameter("tele_exposure_track_offset").value
        )
        self._wide_exposure_gain = float(self.get_parameter("wide_exposure_gain").value)
        self._wide_exposure_offset = float(self.get_parameter("wide_exposure_offset").value)
        self._tele_exposure_lock_frames = max(
            1, int(self.get_parameter("tele_exposure_lock_frames").value)
        )
        self._tele_exposure_lost_frames = max(
            self._tele_exposure_lock_frames,
            int(self.get_parameter("tele_exposure_lost_frames").value),
        )
        self._tele_exposure_min_switch_interval_s = max(
            0.1, float(self.get_parameter("tele_exposure_min_switch_interval_s").value)
        )

        wide_topic = str(self.get_parameter("wide_image_topic").value)
        tele_topic = str(self.get_parameter("tele_image_topic").value)
        upper_cmd_topic = str(self.get_parameter("upper_cmd_topic").value)
        debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        wide_debug_image_topic = str(self.get_parameter("wide_debug_image_topic").value)
        wide_detection_topic = str(self.get_parameter("wide_detection_topic").value)
        _dbg_master = bool(self.get_parameter("enable_debug_image_publish").value)
        self._enable_tele_debug_image = _dbg_master and bool(
            self.get_parameter("enable_tele_debug_image_publish").value
        )
        self._enable_wide_debug_image = (
            _dbg_master
            and bool(self.get_parameter("enable_wide_debug_image_publish").value)
            and self._enable_short_focal_pipeline
        )

        if self._enable_short_focal_pipeline and self._enable_wide_detection:
            self.create_subscription(
                Image, wide_topic, self._on_wide_image, qos_profile_sensor_data
            )
        self.create_subscription(
            Image, tele_topic, self._on_tele_image, qos_profile_sensor_data
        )
        self._setup_tele_raw_recording()
        self._setup_tele_game_folder_images()
        self._upper_pub = self.create_publisher(Float64MultiArray, upper_cmd_topic, 20)
        if self._enable_tele_debug_image:
            self._debug_img_pub = self.create_publisher(Image, debug_image_topic, 5)
        else:
            self._debug_img_pub = None
        if self._enable_short_focal_pipeline:
            if self._enable_wide_debug_image:
                self._wide_debug_img_pub = self.create_publisher(Image, wide_debug_image_topic, 5)
            else:
                self._wide_debug_img_pub = None
            if self._enable_wide_detection:
                self._wide_det_pub = self.create_publisher(
                    Float64MultiArray, wide_detection_topic, 20
                )
            else:
                self._wide_det_pub = None
        else:
            # 总闸关：不注册任何短焦输出话题（无 Publisher），接上短焦相机也不消费其话题
            self._wide_debug_img_pub = None
            self._wide_det_pub = None

        self._async_debug_publish = bool(self.get_parameter("enable_async_debug_publish").value)
        self._async_debug_drop = bool(self.get_parameter("async_debug_drop_if_busy").value)
        self._tele_debug_image_max_width = max(
            0, int(self.get_parameter("tele_debug_image_max_width").value)
        )
        self._wide_debug_image_max_width = max(
            0, int(self.get_parameter("wide_debug_image_max_width").value)
        )
        self._tele_debug_fut: Optional[Future] = None
        self._wide_debug_fut: Optional[Future] = None
        self._debug_pub_executor: Optional[ThreadPoolExecutor] = None
        if self._async_debug_publish and (
            self._enable_tele_debug_image or self._enable_wide_debug_image
        ):
            self._debug_pub_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="lt_dbg",
            )

        hz = max(1.0, float(self.get_parameter("loop_hz").value))
        self.create_timer(1.0 / hz, self._on_timer)
        self._start_detect_workers()
        self._load_boresight_file_if_present()
        if not self._enable_short_focal_pipeline:
            self.get_logger().info(
                "短焦整链已关闭(enable_short_focal_pipeline:=false)：不订阅短焦、不发布任何短焦话题、"
                "不推理短焦；仅长焦。物理短焦接上亦被忽略。hw_sdk 建议 start_wide_camera:=false。"
            )
        rec_tail = ""
        if self._tele_record_thread is not None and self._tele_record_file is not None:
            rec_tail = (
                f", tele_raw_recording=ON file={self._tele_record_file} "
                f"fps={self._tele_record_fps:.1f} fourcc={self._tele_record_fourcc}"
            )
        elif not bool(self.get_parameter("enable_tele_raw_recording").value):
            rec_tail = ", tele_raw_recording=OFF"
        else:
            rec_tail = ", tele_raw_recording=FAILED(见上方错误日志)"
        game_tail = ""
        if self._tele_game_thread is not None and self._tele_game_dir is not None:
            game_tail = (
                f", tele_game_images=ON dir={self._tele_game_dir} "
                f"subsample={self._tele_game_subsample} fmt={self._tele_game_ext}"
            )
        elif not bool(self.get_parameter("enable_tele_game_folder_images").value):
            game_tail = ", tele_game_images=OFF"
        else:
            game_tail = ", tele_game_images=FAILED(见上方错误日志)"
        if self._enable_short_focal_pipeline:
            io_tail = (
                f"wide={wide_topic}, tele={tele_topic}, cmd={upper_cmd_topic}, "
                f"debug={debug_image_topic}, wide_debug={wide_debug_image_topic}, "
                f"wide_det={wide_detection_topic}, runtime_mode={self._runtime_mode}, "
                f"debug_img tele={self._enable_tele_debug_image} wide={self._enable_wide_debug_image}, "
                f"tele_debug_max_w={self._tele_debug_image_max_width}, "
                f"wide_debug_max_w={self._wide_debug_image_max_width}, "
                f"wide_hz_idle={self._wide_detect_hz_idle:.2f}, wide_hz_reacq={self._wide_detect_hz_reacq:.2f}, "
                f"async_debug_publish={self._async_debug_publish}"
                f"{rec_tail}{game_tail}"
            )
        else:
            io_tail = (
                f"tele={tele_topic}, cmd={upper_cmd_topic}, debug={debug_image_topic}, "
                f"runtime_mode={self._runtime_mode}, "
                f"debug_img tele={self._enable_tele_debug_image}, tele_debug_max_w={self._tele_debug_image_max_width}, "
                f"async_debug_publish={self._async_debug_publish} "
                f"(短焦 wide/wide_debug/wide_det 未注册)"
                f"{rec_tail}{game_tail}"
            )
        self.get_logger().info(
            "lasertracking_tracker 启动: "
            f"enable_short_focal_pipeline={self._enable_short_focal_pipeline}, "
            f"enable_wide_detection={self._enable_wide_detection}, "
            f"enable_kalman_filter={self._enable_kalman}, "
            f"enable_reacq_assist={self._enable_reacq_assist}, "
            f"enable_reacq_scan={self._enable_reacq_scan}, "
            f"enable_reacq_wide_assist={self._enable_reacq_wide_assist}, "
            f"reacq_require_first_lock={self._reacq_require_first_lock}, "
            f"{io_tail}"
        )
        self.get_logger().info(
            f"aim_offset(base) dx={self._base_aim_pixel_offset[0]:+.1f}px "
            f"dy={self._base_aim_pixel_offset[1]:+.1f}px source={self._boresight_source}"
        )
        if self._enable_short_focal_pipeline:
            self.get_logger().info(
                "海康曝光/增益(hw_params；hw_sdk 启动时会据此覆盖两路海康节点): "
                f"wide exp={float(self.get_parameter('hw_hik_wide_exposure_time').value):.0f}µs "
                f"gain={float(self.get_parameter('hw_hik_wide_gain').value):.1f}dB; "
                f"tele exp={float(self.get_parameter('hw_hik_tele_exposure_time').value):.0f}µs "
                f"gain={float(self.get_parameter('hw_hik_tele_gain').value):.1f}dB"
            )
        else:
            self.get_logger().info(
                "海康曝光/增益(仅长焦节点；短焦整链已关，以下仅 tele 覆盖值): "
                f"tele exp={float(self.get_parameter('hw_hik_tele_exposure_time').value):.0f}µs "
                f"gain={float(self.get_parameter('hw_hik_tele_gain').value):.1f}dB"
            )

    def _switch_tele_exposure_mode(self, mode: str) -> None:
        if mode not in ("search", "track"):
            return
        if mode == self._tele_exposure_mode:
            return
        now = time.time()
        if (now - self._last_exposure_switch_ts) < self._tele_exposure_min_switch_interval_s:
            return
        self._tele_exposure_mode = mode
        self._last_exposure_switch_ts = now

    def _update_exposure_state(self, has_tele_detection: bool) -> None:
        if has_tele_detection:
            self._tele_detect_streak += 1
            self._tele_lost_streak = 0
            if (
                self._enable_dynamic_tele_exposure
                and (not self._reacq_active)
                and self._tele_detect_streak >= self._tele_exposure_lock_frames
            ):
                self._switch_tele_exposure_mode("track")
        else:
            self._tele_lost_streak += 1
            self._tele_detect_streak = 0
            if (
                self._enable_dynamic_tele_exposure
                and self._tele_lost_streak >= self._tele_exposure_lost_frames
            ):
                self._switch_tele_exposure_mode("search")

    def _resolve_reacq_scan_center(self) -> tuple[float, float, str]:
        """螺旋扫描原点：优先「丢失过多帧」YAML 预设，其次丢失锚点，再次无锚点时可选预设，否则当前姿态。"""
        py = float(self._reacq_scan_preset_yaw)
        pp = float(self._reacq_scan_preset_pitch)
        thr = int(self._reacq_scan_use_preset_after_total_lost_frames)
        if thr > 0 and self._tele_lost_count >= thr:
            return py, pp, f"YAML预设(相对初始绝对角，丢失帧≥{thr})"
        if self._tele_loss_anchor_valid:
            return (
                float(self._tele_loss_anchor_yaw),
                float(self._tele_loss_anchor_pitch),
                "长焦丢失点(由有→无首帧姿态)",
            )
        if self._reacq_scan_use_preset_without_anchor:
            return py, pp, "YAML预设(相对初始绝对角，无有效丢失锚点)"
        return float(self._yaw), float(self._pitch), "当前姿态(无锚点且未启用预设时回退)"

    def _maybe_reacq_reloc_scan_center_to_preset_for_excessive_loss(self) -> None:
        """已在重获扫描中、长焦连续丢失超过阈值时，将原点迁至 YAML 预设并重卷螺旋角。"""
        thr = int(self._reacq_scan_use_preset_after_total_lost_frames)
        if thr <= 0 or self._tele_lost_count < thr:
            return
        py = float(self._reacq_scan_preset_yaw)
        pp = float(self._reacq_scan_preset_pitch)
        if abs(self._reacq_scan_center_yaw - py) <= 1e-9 and abs(self._reacq_scan_center_pitch - pp) <= 1e-9:
            return
        self._reacq_scan_center_yaw = py
        self._reacq_scan_center_pitch = pp
        self._reacq_scan_theta = 0.0
        self.get_logger().info(
            "丢失扫描原点切换为 YAML 预设: "
            f"yaw={math.degrees(py):.3f}° pitch={math.degrees(pp):.3f}° (长焦连续丢失 {self._tele_lost_count}>={thr})"
        )

    def _start_reacq(self) -> None:
        if self._reacq_active:
            return
        self._reacq_active = True
        self._reacq_seen_count = 0
        self._reacq_scan_theta = 0.0
        cy, cp, anchor_src = self._resolve_reacq_scan_center()
        self._reacq_scan_center_yaw = cy
        self._reacq_scan_center_pitch = cp
        self._reacq_last_wide_err = None
        self._reacq_wide_miss_count = 0
        self._reacq_scan_frames = 0
        self._reacq_wide_frames = 0
        self._reacq_phase = "scan"
        self._wide_detect_hz_current = self._wide_detect_hz_reacq
        self._wide_hz_mode = "reacq"
        self.get_logger().info(
            "进入辅助重获模式: 螺旋扫描原点="
            f"yaw={math.degrees(self._reacq_scan_center_yaw):.3f}° pitch={math.degrees(self._reacq_scan_center_pitch):.3f}° ({anchor_src})；"
            "短焦阶段回到扫描时不移动原点"
        )

    def _stop_reacq(self, reason: str) -> None:
        if self._reacq_active:
            self.get_logger().info(f"退出辅助重获模式: {reason}")
        self._reacq_active = False
        self._reacq_seen_count = 0
        self._reacq_scan_theta = 0.0
        self._reacq_last_wide_err = None
        self._reacq_wide_miss_count = 0
        self._reacq_scan_frames = 0
        self._reacq_wide_frames = 0
        self._reacq_phase = "scan"
        self._wide_detect_hz_current = self._wide_detect_hz_idle
        self._wide_hz_mode = "idle"
        self._tele_loss_anchor_valid = False

    def _publish_current_pose_command(self) -> None:
        self._emit_pose_command(0.0, 0.0)

    def _run_reacq_wide_assist(self, yaw_err: float, pitch_err: float) -> None:
        # 以“双相机初始位姿参考”作为固定基准，将短焦误差角映射到长焦目标角。
        yaw_target = self._reacq_wide_ref_yaw + yaw_err
        pitch_target = self._reacq_wide_ref_pitch + pitch_err
        yaw_min, yaw_max = self._get_yaw_clamp_range()
        pitch_lim = abs(float(self.get_parameter("pitch_limit_rad").value))
        yaw_target = self._clamp(yaw_target, yaw_min, yaw_max)
        pitch_target = self._clamp(pitch_target, -pitch_lim, pitch_lim)
        max_step = max(1e-3, float(self._reacq_max_step))
        dy = self._clamp(yaw_target - self._yaw, -max_step, max_step)
        dp = self._clamp(pitch_target - self._pitch, -max_step, max_step)
        self._yaw += dy
        self._pitch += dp

        self._publish_current_pose_command()

    def _map_wide_error_to_tele_error(self, yaw_err: float, pitch_err: float) -> tuple[float, float]:
        """短焦误差角映射到长焦控制误差角（2x2可调矩阵）。"""
        mapped_yaw = self._reacq_wide_map_m00 * yaw_err + self._reacq_wide_map_m01 * pitch_err
        mapped_pitch = self._reacq_wide_map_m10 * yaw_err + self._reacq_wide_map_m11 * pitch_err
        return float(mapped_yaw), float(mapped_pitch)

    def _reacq_spiral_r_and_dr_dtheta(self, theta: float) -> tuple[float, float]:
        """分段阿基米德螺旋半径 r(θ) 与 dr/dθ；内圈小步进压实中心，θ1 后接外层 spacing。"""
        r_max = float(self._reacq_scan_spiral_r_max)
        a_in = float(self._reacq_scan_spiral_a_inner)
        a_out = float(self._reacq_scan_spiral_a_outer)
        th1 = float(self._reacq_scan_spiral_theta1)
        if th1 <= 0.0:
            r = min(a_out * theta, r_max)
            dr = a_out if r < r_max else 0.0
            return r, dr
        r1 = min(a_in * th1, r_max)
        if theta <= th1:
            r = min(a_in * theta, r_max)
            dr = a_in if r < r_max else 0.0
            return r, dr
        r = min(r1 + a_out * (theta - th1), r_max)
        dr = a_out if r < r_max else 0.0
        return r, dr

    def _run_reacq_scan_fallback(self, dt: float) -> None:
        if not self._reacq_scan_fallback:
            return
        self._maybe_reacq_reloc_scan_center_to_preset_for_excessive_loss()
        self._reacq_scan_frames += 1
        theta = float(self._reacq_scan_theta)
        speed = float(self._reacq_scan_spiral_speed)
        r_max = float(self._reacq_scan_spiral_r_max)
        yaw_scale = float(self._reacq_scan_spiral_yaw_scale)
        pitch_scale = float(self._reacq_scan_spiral_pitch_scale)
        r, dr_dtheta = self._reacq_spiral_r_and_dr_dtheta(theta)
        # 在 (yaw,pitch) 平面上阿基米德螺旋：theta 扫一圈时 cos/sin 使四向均覆盖；两轴尺度建议相同以呈圆盘而非椭圆。
        dx_dtheta = yaw_scale * (dr_dtheta * math.cos(theta) - r * math.sin(theta))
        dy_dtheta = pitch_scale * (dr_dtheta * math.sin(theta) + r * math.cos(theta))
        ds_dtheta = max(1e-5, math.hypot(dx_dtheta, dy_dtheta))
        theta += (speed / ds_dtheta) * dt
        # 到达上限后回卷，持续在局部区域重复搜索而不是停住。
        if theta >= self._reacq_scan_spiral_theta_max:
            theta = 0.0
        self._reacq_scan_theta = theta
        r_eval, _ = self._reacq_spiral_r_and_dr_dtheta(theta)
        yaw_offset = yaw_scale * r_eval * math.cos(theta)
        pitch_offset = pitch_scale * r_eval * math.sin(theta)
        yaw_min, yaw_max = self._get_yaw_clamp_range()
        pitch_lim = abs(float(self.get_parameter("pitch_limit_rad").value))
        self._yaw = self._clamp(self._reacq_scan_center_yaw + yaw_offset, yaw_min, yaw_max)
        self._pitch = self._clamp(
            self._reacq_scan_center_pitch + pitch_offset, -pitch_lim, pitch_lim
        )
        self._emit_pose_command(0.0, 0.0)

    def _run_reacq_assist(
        self,
        wide_det: Optional[Detection],
        wide_size: Optional[Tuple[int, int]],
        dt: float,
    ) -> str:
        # 循环重获：扫描N帧 -> 短焦辅助M帧 -> 扫描N帧 ...
        # 避免短焦映射误差导致一直卡在同一错误位置。
        allow_wide = self._enable_reacq_wide_assist and self._reacq_use_wide and self._enable_wide_detection
        allow_scan = self._enable_reacq_scan and self._reacq_scan_fallback

        if self._reacq_phase == "scan":
            if allow_scan:
                self._run_reacq_scan_fallback(dt)
                if allow_wide and (
                    self._tele_lost_count >= self._reacq_wide_enter_lost_frames
                    or self._reacq_scan_frames >= self._reacq_scan_to_wide_frames
                ):
                    self._reacq_phase = "wide"
                    self._reacq_wide_frames = 0
                    self._reacq_wide_miss_count = 0
                    self._reacq_last_wide_err = None
                return "SCAN_STAGE"
            if allow_wide:
                self._reacq_phase = "wide"
            else:
                return "HOLD_STAGE"

        if (
            allow_wide
            and wide_det is not None
            and wide_size is not None
        ):
            self._reacq_wide_frames += 1
            ww, wh = wide_size
            werr_x = (wide_det.cx - 0.5 * ww) / max(1.0, 0.5 * ww)
            werr_y = (wide_det.cy - 0.5 * wh) / max(1.0, 0.5 * wh)
            yaw_sign = float(self.get_parameter("yaw_error_sign").value)
            pitch_sign = float(self.get_parameter("pitch_error_sign").value)
            # 针孔相机精确公式: θ = atan(norm_offset * tan(hfov/2))
            yaw_err = yaw_sign * math.atan(werr_x * math.tan(self._wide_hfov * 0.5))
            pitch_err = pitch_sign * math.atan((-werr_y) * math.tan(self._wide_vfov * 0.5))
            yaw_err, pitch_err = self._map_wide_error_to_tele_error(yaw_err, pitch_err)
            self._reacq_last_wide_err = (yaw_err, pitch_err)
            self._reacq_wide_miss_count = 0

            self._run_reacq_wide_assist(yaw_err, pitch_err)
            if self._reacq_wide_frames >= self._reacq_wide_phase_frames:
                self._reacq_phase = "scan"
                self._reacq_scan_theta = 0.0
                self._reacq_scan_frames = 0
                self._reacq_wide_frames = 0
            return "WIDE_ASSIST"
        if allow_wide and self._reacq_last_wide_err is not None:
            self._reacq_wide_frames += 1
            self._reacq_wide_miss_count += 1
            if self._reacq_wide_miss_count <= self._reacq_wide_lost_tolerance:
                ly, lp = self._reacq_last_wide_err
                self._run_reacq_wide_assist(ly, lp)
                if self._reacq_wide_frames >= self._reacq_wide_phase_frames:
                    self._reacq_phase = "scan"
                    self._reacq_scan_theta = 0.0
                    self._reacq_scan_frames = 0
                    self._reacq_wide_frames = 0
                return "WIDE_HOLD"

        if allow_wide and self._reacq_wide_frames >= self._reacq_wide_phase_frames:
            self._reacq_phase = "scan"
            self._reacq_scan_theta = 0.0
            self._reacq_scan_frames = 0
            self._reacq_wide_frames = 0

        # 短焦当前不可用时继续局部扫描，确保不会“失败后停住不作为”。
        if allow_scan:
            self._run_reacq_scan_fallback(dt)
            return "SCAN_WAIT_WIDE"
        return "HOLD_WAIT_WIDE"

    def _resolve_model_path(self, raw_path: str) -> Optional[str]:
        raw = os.path.expanduser(raw_path.strip())
        candidates = []

        # 原样（绝对路径或当前目录相对路径）
        candidates.append(Path(raw))

        # 常见误写: /lasertracking/model/aerial.pt（本意是仓库内相对路径）
        if raw.startswith("/lasertracking/"):
            candidates.append(Path.cwd() / raw.lstrip("/"))

        # 再尝试仓库根（当前脚本位于 <repo>/radar_gimbal_gazebo/scripts）
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parents[2] if len(script_dir.parents) >= 3 else Path.cwd()
        candidates.append(repo_root / raw.lstrip("/"))

        seen = set()
        for p in candidates:
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            if Path(rp).is_file():
                return rp
        return None

    def _resolve_weight_for_branch(self, pt_or_weights: str, branch_trt_override: str) -> Optional[str]:
        """解析长/短焦实际加载的权重路径（.pt 或 .engine）。"""
        use_tensorrt = bool(self.get_parameter("use_tensorrt").value)
        default_trt = str(self.get_parameter("trt_engine_path").value).strip()
        ovr = str(branch_trt_override).strip()
        if use_tensorrt:
            for cand in (ovr, default_trt):
                if cand:
                    r = self._resolve_model_path(cand)
                    if r:
                        return r
            if pt_or_weights.endswith(".pt"):
                r = self._resolve_model_path(pt_or_weights[:-3] + ".engine")
                if r:
                    return r
        if pt_or_weights.endswith(".engine"):
            r = self._resolve_model_path(pt_or_weights)
            if r:
                return r
        return self._resolve_model_path(pt_or_weights)

    def _resolve_repo_path(self, raw_path: str) -> Optional[Path]:
        p = str(raw_path).strip()
        if not p:
            return None
        cand = Path(os.path.expanduser(p))
        if cand.is_absolute():
            return cand
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parents[2] if len(script_dir.parents) >= 3 else Path.cwd()
        c1 = repo_root / p
        if c1.exists():
            return c1
        return Path.cwd() / p

    def _load_boresight_file_if_present(self) -> None:
        raw = str(self.get_parameter("boresight_file").value).strip()
        if not raw:
            return
        path = self._resolve_repo_path(raw)
        if path is None or (not path.exists()):
            self.get_logger().info(f"boresight 文件不存在，回退 pixel_offset: {raw}")
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if "u_L" not in data or "v_L" not in data:
                self.get_logger().warn(f"boresight 文件缺少 u_L/v_L，回退 pixel_offset: {path}")
                return
            u_l = float(data["u_L"])
            v_l = float(data["v_L"])
            img_w = int(data.get("image_w", 0) or 0)
            img_h = int(data.get("image_h", 0) or 0)
            if img_w > 0 and img_h > 0:
                self._base_aim_pixel_offset = (u_l - 0.5 * img_w, v_l - 0.5 * img_h)
                self._aim_pixel_offset = self._base_aim_pixel_offset
                self._boresight_source = "file"
            if "bx" in data:
                self._parallax_baseline_bx = float(data["bx"])
            if "by" in data:
                self._parallax_baseline_by = float(data["by"])
            if "z_ref" in data:
                self._parallax_z_ref = float(data["z_ref"])
            self.get_logger().info(
                f"已加载 boresight: {path} dx={self._base_aim_pixel_offset[0]:+.1f}px "
                f"dy={self._base_aim_pixel_offset[1]:+.1f}px"
            )
        except Exception as exc:
            self.get_logger().warn(f"boresight 加载失败，回退 pixel_offset: {exc}")

    def _normalize_distance(self, distance_m: Optional[float]) -> Optional[float]:
        if distance_m is None:
            return None
        try:
            value = float(distance_m)
        except (TypeError, ValueError):
            return None
        if (not np.isfinite(value)) or value <= 0.0:
            return None
        return min(self._parallax_max_distance_m, max(self._parallax_min_distance_m, value))

    def _estimate_distance_from_bbox(self, det: Optional[Detection]) -> tuple[Optional[float], str]:
        if det is None:
            return None, "bbox"
        x1, y1, x2, y2 = det.xyxy
        bw = max(0.0, float(x2 - x1))
        bh = max(0.0, float(y2 - y1))
        if max(bw, bh) < self._parallax_min_bbox_px:
            return None, "bbox_small"
        width_distance = None
        height_distance = None
        if self._parallax_target_width_m > 0.0 and bw >= self._parallax_min_bbox_px and self._calibration_fx > 0.0:
            width_distance = self._calibration_fx * self._parallax_target_width_m / bw
        if self._parallax_target_height_m > 0.0 and bh >= self._parallax_min_bbox_px and self._calibration_fy > 0.0:
            height_distance = self._calibration_fy * self._parallax_target_height_m / bh

        if self._parallax_source == "bbox_width":
            return self._normalize_distance(width_distance), "bbox_w"
        if self._parallax_source == "bbox_height":
            return self._normalize_distance(height_distance), "bbox_h"
        candidates = [v for v in (width_distance, height_distance) if v is not None]
        if not candidates:
            return None, "bbox_size"
        if len(candidates) == 1:
            return self._normalize_distance(candidates[0]), ("bbox_w" if width_distance is not None else "bbox_h")
        return self._normalize_distance(sum(candidates) / len(candidates)), "bbox_wh"

    def _resolve_target_distance(self, det: Optional[Detection]) -> tuple[Optional[float], str]:
        if not self._parallax_enabled:
            return None, "off"
        fixed_distance = self._normalize_distance(self._parallax_fixed_distance_m)
        if self._parallax_source == "fixed":
            return fixed_distance, "fixed"
        if self._parallax_source in ("bbox_width", "bbox_height", "bbox_auto", "auto"):
            bbox_d, bbox_s = self._estimate_distance_from_bbox(det)
            if bbox_d is not None:
                return bbox_d, bbox_s
            if fixed_distance is not None:
                return fixed_distance, "fixed"
            return None, bbox_s
        return fixed_distance, "fixed"

    def _compute_parallax_residual(self, distance_m: Optional[float]) -> tuple[float, float]:
        d = self._normalize_distance(distance_m)
        if d is None:
            return 0.0, 0.0
        if abs(self._parallax_baseline_bx) < 1e-9 and abs(self._parallax_baseline_by) < 1e-9:
            return 0.0, 0.0
        if self._calibration_fx <= 0.0 or self._calibration_fy <= 0.0:
            return 0.0, 0.0
        du = self._calibration_fx * self._parallax_baseline_bx / d
        dv = self._calibration_fy * self._parallax_baseline_by / d
        if self._parallax_z_ref > 0.0:
            du -= self._calibration_fx * self._parallax_baseline_bx / self._parallax_z_ref
            dv -= self._calibration_fy * self._parallax_baseline_by / self._parallax_z_ref
        return float(du), float(dv)

    def _update_effective_aim_pixel_offset(self, det: Optional[Detection]) -> None:
        distance_m, source = self._resolve_target_distance(det)
        if distance_m is not None:
            if self._parallax_last_distance_m is None or self._parallax_distance_alpha >= 1.0:
                smoothed = distance_m
            elif self._parallax_distance_alpha <= 0.0:
                smoothed = self._parallax_last_distance_m
            else:
                smoothed = (
                    self._parallax_distance_alpha * distance_m
                    + (1.0 - self._parallax_distance_alpha) * self._parallax_last_distance_m
                )
            self._parallax_last_distance_m = self._normalize_distance(smoothed)
            self._parallax_last_distance_source = source
        elif not self._parallax_hold_last_distance:
            self._parallax_last_distance_m = None
            self._parallax_last_distance_source = source

        pdx, pdy = self._compute_parallax_residual(self._parallax_last_distance_m)
        self._parallax_last_offset = (pdx, pdy)
        self._aim_pixel_offset = (
            self._base_aim_pixel_offset[0] + pdx,
            self._base_aim_pixel_offset[1] + pdy,
        )

    @staticmethod
    def _tele_recording_suffix_for_fourcc(fourcc: str, extension_override: str) -> str:
        o = str(extension_override).strip().lower()
        if o:
            return o if o.startswith(".") else f".{o}"
        fc = str(fourcc).strip().upper()
        if fc in ("MJPG", "JPEG", "IJPG", "XVID", "DIVX", "DX50", "FMP4", "HFYU"):
            return ".avi"
        if fc in ("MP4V", "H264", "X264", "AVC1", "HEVC", "H265"):
            return ".mp4"
        return ".avi"

    def _try_open_tele_video_writer(self, w0: int, h0: int) -> tuple[Optional[cv2.VideoWriter], Path, str]:
        """依次尝试主 fourcc 与常见回退，解决部分系统 MJPG 不可用导致零文件的问题。"""
        fps = float(self._tele_record_fps)
        base = self._tele_record_file
        if base is None:
            return None, Path("."), ""
        primary = (self._tele_record_fourcc or "MJPG").strip()
        if len(primary) != 4:
            primary = "MJPG"
        candidates: list[str] = []
        for c in (primary.upper(), "XVID", "MJPG", "MP4V"):
            if len(c) == 4 and c not in candidates:
                candidates.append(c)
        last_path = base
        for idx, fc in enumerate(candidates):
            suf = self._tele_recording_suffix_for_fourcc(fc, "")
            if idx == 0:
                path = base
            else:
                path = base.parent / f"{base.stem}_{fc.lower()}{suf}"
            last_path = path
            vw = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*fc), fps, (int(w0), int(h0))
            )
            if vw.isOpened():
                if idx > 0:
                    self.get_logger().warning(
                        f"长焦内录主编码 {primary} 打开失败，已改用 {fc} 写入 {path}"
                    )
                self._tele_record_file = path
                self._tele_record_fourcc = fc
                return vw, path, fc
            try:
                if path.is_file() and path.stat().st_size == 0:
                    path.unlink(missing_ok=True)
            except Exception:
                pass
        self.get_logger().error(
            f"长焦内录无法打开 VideoWriter（已依次尝试 {candidates}），最后路径={last_path}。"
            "请检查 OpenCV 视频后端、磁盘权限；NTFS/exFAT 上可优先试 XVID+.avi。"
        )
        return None, last_path, primary

    def _resolve_tele_recording_root(self, *, for_what: str) -> Optional[Path]:
        """长焦落盘根目录：非空 tele_recording_output_dir 须已存在；否则用 cwd/recordings/tele_raw 并自动创建。"""
        raw_dir = str(self.get_parameter("tele_recording_output_dir").value).strip()
        if raw_dir:
            root = Path(raw_dir).expanduser().resolve()
            if not root.is_dir():
                self.get_logger().error(
                    f"{for_what}: 输出目录不存在或不是目录: {root}。"
                    "非空 tele_recording_output_dir 须在启动前已存在并可写。"
                )
                return None
        else:
            root = (Path.cwd() / "recordings" / "tele_raw").resolve()
            try:
                root.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self.get_logger().error(f"{for_what}: 默认输出目录创建失败: {root} err={exc}")
                return None
        try:
            probe = root / ".lasertracking_tele_recording_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except Exception as exc:
            self.get_logger().error(
                f"{for_what}: 目录不可写（外置盘是否已挂载？是否只读？）: {root} err={exc}"
            )
            return None
        return root

    def _setup_tele_raw_recording(self) -> None:
        if not bool(self.get_parameter("enable_tele_raw_recording").value):
            return
        root = self._resolve_tele_recording_root(for_what="长焦内录")
        if root is None:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        fc = str(self.get_parameter("tele_recording_fourcc").value).strip()
        if len(fc) != 4:
            self.get_logger().warn("tele_recording_fourcc 须为 4 字符，已回退 MJPG（AVI 帧内编码，利于实时）")
            fc = "MJPG"
        ext_ov = str(self.get_parameter("tele_recording_extension").value).strip()
        suffix = self._tele_recording_suffix_for_fourcc(fc, ext_ov)
        out_path = root / f"tele_long_{ts}{suffix}"
        qmax = max(4, int(self.get_parameter("tele_recording_queue_max").value))
        fps = max(1.0, float(self.get_parameter("tele_recording_fps").value))

        self._tele_record_file = out_path
        self._tele_record_fps = fps
        self._tele_record_fourcc = fc
        self._tele_record_subsample = max(1, int(self.get_parameter("tele_recording_subsample").value))
        self._tele_record_subsample_counter = 0
        self._tele_record_queue = queue.Queue(maxsize=qmax)
        self._tele_record_running = True
        self._tele_record_thread = threading.Thread(
            target=self._tele_raw_recording_worker,
            name="tele_raw_recorder",
            daemon=True,
        )
        self._tele_record_thread.start()
        self.get_logger().info(
            f"长焦原图内录已启用: 绝对路径={root}；回调仅拷贝原始图像字节入队，解码/编码在后台线程；"
            f"计划文件 {out_path.name} (fps={fps:.1f}, fourcc={fc}, queue_max={qmax}, subsample={self._tele_record_subsample})"
        )

    def _stop_tele_raw_recording(self) -> None:
        th = self._tele_record_thread
        q = self._tele_record_queue
        if th is None or q is None:
            return
        self._tele_record_running = False
        lim = int(q.maxsize) if int(q.maxsize) > 0 else 64
        for _ in range(max(8, lim)):
            try:
                q.put_nowait(None)
                break
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
        th.join(timeout=15.0)
        if th.is_alive():
            self.get_logger().warn("长焦内录线程未在超时内结束，文件可能不完整")
        self._tele_record_thread = None
        self._tele_record_queue = None

    def _setup_tele_game_folder_images(self) -> None:
        if not bool(self.get_parameter("enable_tele_game_folder_images").value):
            return
        root = self._resolve_tele_recording_root(for_what="长焦按帧存图(game 文件夹)")
        if root is None:
            return
        prefix = str(self.get_parameter("tele_game_folder_prefix").value).strip() or "game"
        now = time.time()
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        us = int((now - int(now)) * 1_000_000)
        game_dir = root / f"{prefix}_{ts}_{us:06d}"
        try:
            game_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            game_dir = root / f"{prefix}_{ts}_{us:06d}_{int(time.time_ns()) % 1_000_000_000:09d}"
            try:
                game_dir.mkdir(parents=False, exist_ok=False)
            except Exception as exc:
                self.get_logger().error(f"长焦 game 子目录创建失败: {game_dir} err={exc}")
                return
        except Exception as exc:
            self.get_logger().error(f"长焦 game 子目录创建失败: {game_dir} err={exc}")
            return

        fmt = str(self.get_parameter("tele_game_image_format").value).strip().lower()
        if fmt in ("jpeg", "jpg", ".jpg", ".jpeg"):
            ext = ".jpg"
        elif fmt in ("png", ".png"):
            ext = ".png"
        else:
            self.get_logger().warn(f"tele_game_image_format 未知值 {fmt!r}，已回退 jpg")
            ext = ".jpg"
        qmax = max(4, int(self.get_parameter("tele_game_image_queue_max").value))
        subs = max(1, int(self.get_parameter("tele_game_image_subsample").value))
        jq = int(self.get_parameter("tele_game_jpeg_quality").value)
        jq = max(1, min(100, jq))

        self._tele_game_dir = game_dir
        self._tele_game_ext = ext
        self._tele_game_jpeg_quality = jq
        self._tele_game_subsample = subs
        self._tele_game_subsample_counter = 0
        self._tele_game_queue = queue.Queue(maxsize=qmax)
        self._tele_game_running = True
        self._tele_game_queue_full_warned = False
        self._tele_game_decode_fail = 0
        self._tele_game_logged_first = False
        self._tele_game_thread = threading.Thread(
            target=self._tele_game_image_worker,
            name="tele_game_images",
            daemon=True,
        )
        self._tele_game_thread.start()
        self.get_logger().info(
            f"长焦按帧存图已启用: 本次运行目录 {game_dir}（订阅回调仅拷贝原始字节，解码与 imwrite 在后台线程）"
        )

    def _stop_tele_game_folder_images(self) -> None:
        th = self._tele_game_thread
        q = self._tele_game_queue
        if th is None or q is None:
            return
        self._tele_game_running = False
        lim = int(q.maxsize) if int(q.maxsize) > 0 else 64
        for _ in range(max(8, lim)):
            try:
                q.put_nowait(None)
                break
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
        th.join(timeout=30.0)
        if th.is_alive():
            self.get_logger().warn("长焦按帧存图线程未在超时内结束，部分图片可能未写入")
        self._tele_game_thread = None
        self._tele_game_queue = None
        self._tele_game_dir = None

    def _tele_game_image_worker(self) -> None:
        q = self._tele_game_queue
        out_dir = self._tele_game_dir
        if q is None or out_dir is None:
            return
        ext = self._tele_game_ext
        jq = int(self._tele_game_jpeg_quality)
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), jq]
        saved = 0
        while True:
            try:
                item = q.get(timeout=0.25)
            except queue.Empty:
                if not self._tele_game_running and q.empty():
                    break
                continue
            if item is None:
                break
            if not isinstance(item, tuple) or len(item) != 5:
                continue
            raw_b, enc_t, wi, hi, st_t = item
            arr = np.frombuffer(raw_b, dtype=np.uint8)
            st_use = int(st_t) if int(st_t) > 0 else None
            bgr = self._numpy_payload_to_bgr(str(enc_t), int(wi), int(hi), arr, st_use)
            if bgr is None:
                self._tele_game_decode_fail += 1
                if self._tele_game_decode_fail <= 3 or self._tele_game_decode_fail % 150 == 0:
                    self.get_logger().warning(
                        f"长焦存图后台解码失败（累计 {self._tele_game_decode_fail} 次）"
                        f"encoding={enc_t} {wi}x{hi}"
                    )
                continue
            if bgr.size == 0 or bgr.ndim != 3 or bgr.shape[2] != 3:
                continue
            if not self._tele_game_logged_first:
                self._tele_game_logged_first = True
                self.get_logger().info(
                    f"长焦存图已收到可写首帧 encoding={enc_t} {wi}x{hi} -> {out_dir}"
                )
            fn = out_dir / f"frame_{saved:08d}{ext}"
            try:
                if ext == ".jpg":
                    ok = bool(cv2.imwrite(str(fn), bgr, jpeg_params))
                else:
                    ok = bool(cv2.imwrite(str(fn), bgr))
            except Exception:
                ok = False
            if ok:
                saved += 1
            elif saved == 0:
                self.get_logger().warning(f"长焦存图 imwrite 失败: {fn}")
        self.get_logger().info(f"长焦按帧存图结束: 目录 {out_dir} 共写入约 {saved} 张")

    def _tele_raw_recording_worker(self) -> None:
        q = self._tele_record_queue
        if q is None:
            return
        writer: Optional[cv2.VideoWriter] = None
        out_path = self._tele_record_file
        if out_path is None:
            return
        w0 = h0 = 0
        frames = 0
        used_disk_path: Optional[Path] = None
        while True:
            try:
                item = q.get(timeout=0.25)
            except queue.Empty:
                if not self._tele_record_running and q.empty():
                    break
                continue
            if item is None:
                break
            if not isinstance(item, tuple) or len(item) != 5:
                continue
            raw_b, enc_t, wi, hi, st_t = item
            arr = np.frombuffer(raw_b, dtype=np.uint8)
            st_use = int(st_t) if int(st_t) > 0 else None
            bgr = self._numpy_payload_to_bgr(str(enc_t), int(wi), int(hi), arr, st_use)
            if bgr is None:
                self._tele_record_decode_fail += 1
                if self._tele_record_decode_fail <= 3 or self._tele_record_decode_fail % 150 == 0:
                    self.get_logger().warning(
                        f"长焦内录后台解码失败（累计 {self._tele_record_decode_fail} 次）"
                        f"encoding={enc_t} {wi}x{hi} step={st_t}；请核对相机 encoding。"
                    )
                continue
            bgr = np.ascontiguousarray(bgr)
            if not self._tele_record_logged_first_frame:
                self._tele_record_logged_first_frame = True
                self.get_logger().info(
                    f"长焦内录已收到可写首帧（后台解码）encoding={enc_t} {wi}x{hi}"
                )
            if bgr.size == 0 or bgr.ndim != 3 or bgr.shape[2] != 3:
                continue
            h, w = bgr.shape[:2]
            if writer is None:
                w0, h0 = int(w), int(h)
                writer, opened_path, used_fc = self._try_open_tele_video_writer(w0, h0)
                if writer is None:
                    self._tele_record_running = False
                    while True:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                    return
                used_disk_path = opened_path
                out_path = opened_path
                self.get_logger().info(
                    f"长焦内录开始写入视频: {opened_path} 分辨率 {w0}x{h0} fourcc={used_fc}"
                )
            if int(w) != w0 or int(h) != h0:
                continue
            writer.write(bgr)
            frames += 1
        if writer is not None:
            writer.release()
            fin = used_disk_path if used_disk_path is not None else out_path
            if frames > 0:
                self.get_logger().info(f"长焦内录已保存: {fin}（约 {frames} 帧）")
            else:
                try:
                    fin.unlink(missing_ok=True)
                except Exception:
                    pass
                self.get_logger().warn(
                    f"长焦内录无有效帧，已删除空文件: {fin}。"
                    "若全程无「已收到可写首帧」日志，请检查长焦话题是否有图、编码是否受支持。"
                )
        else:
            self.get_logger().warn(
                "长焦内录线程退出但未创建视频文件：运行期间未收到可解码图像，或节点在首帧到达前已退出。"
                "请确认 tele_image_topic 有发布、外置盘路径已挂载且可写。"
            )

    def _enqueue_tele_raw_frame(self, msg: Image) -> None:
        """仅拷贝 sensor_msgs/Image 原始 payload 入队；不在此处做 OpenCV 解码，避免拖慢订阅回调与图像链路。"""
        if not self._tele_record_running or self._tele_record_queue is None:
            return
        self._tele_record_subsample_counter += 1
        if self._tele_record_subsample > 1:
            if (self._tele_record_subsample_counter % self._tele_record_subsample) != 0:
                return
        w, h = int(msg.width), int(msg.height)
        if w <= 0 or h <= 0:
            return
        st = int(getattr(msg, "step", 0) or 0)
        enc = str(msg.encoding)
        try:
            raw_b = bytes(msg.data)
        except Exception:
            return
        if not raw_b:
            return
        payload = (raw_b, enc, w, h, st)
        try:
            self._tele_record_queue.put_nowait(payload)
        except queue.Full:
            if not self._tele_record_queue_full_warned:
                self.get_logger().warning(
                    "长焦内录队列已满，将丢弃最旧帧以跟上实时流；若频繁出现可增大 tele_recording_queue_max 或 tele_recording_subsample"
                )
                self._tele_record_queue_full_warned = True
            try:
                _ = self._tele_record_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._tele_record_queue.put_nowait(payload)
            except queue.Full:
                pass

    def _enqueue_tele_game_image(self, msg: Image) -> None:
        if not self._tele_game_running or self._tele_game_queue is None:
            return
        self._tele_game_subsample_counter += 1
        if self._tele_game_subsample > 1:
            if (self._tele_game_subsample_counter % self._tele_game_subsample) != 0:
                return
        w, h = int(msg.width), int(msg.height)
        if w <= 0 or h <= 0:
            return
        st = int(getattr(msg, "step", 0) or 0)
        enc = str(msg.encoding)
        try:
            raw_b = bytes(msg.data)
        except Exception:
            return
        if not raw_b:
            return
        payload = (raw_b, enc, w, h, st)
        try:
            self._tele_game_queue.put_nowait(payload)
        except queue.Full:
            if not self._tele_game_queue_full_warned:
                self.get_logger().warning(
                    "长焦存图队列已满，将丢弃最旧帧；可增大 tele_game_image_queue_max 或 tele_game_image_subsample"
                )
                self._tele_game_queue_full_warned = True
            try:
                _ = self._tele_game_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._tele_game_queue.put_nowait(payload)
            except queue.Full:
                pass

    def _on_wide_image(self, msg: Image) -> None:
        # 防御：无订阅时不会触发；若将来动态改参等导致回调仍被调用，直接丢弃短焦帧
        if not self._enable_short_focal_pipeline:
            return
        self._wide_latest = msg

    def _on_tele_image(self, msg: Image) -> None:
        self._tele_latest = msg
        self._enqueue_tele_raw_frame(msg)
        self._enqueue_tele_game_image(msg)

    def _apply_exposure_preprocess(
        self, frame: np.ndarray, gain: float, offset: float
    ) -> np.ndarray:
        g = float(max(0.1, min(4.0, gain)))
        b = float(max(-80.0, min(80.0, offset)))
        if abs(g - 1.0) < 1e-6 and abs(b) < 1e-6:
            return frame
        return cv2.convertScaleAbs(frame, alpha=g, beta=b)

    def _resolve_lasertracking_dir(self) -> Path:
        """返回 lasertracking 目录路径。优先通过 serial_comm.py 定位。"""
        candidates: list[Path] = []
        try:
            from ament_index_python.packages import get_package_share_directory

            share = Path(get_package_share_directory("radar_gimbal_gazebo")).resolve()
            for base in [share, *share.parents]:
                cand = base / "lasertracking"
                if cand.is_dir() and (
                    (cand / "serial_comm.py").is_file()
                    or (cand / "tracking_system_old" / "serial_comm.py").is_file()
                ):
                    candidates.append(cand)
                    break
        except Exception:
            pass
        script = Path(__file__).resolve()
        for depth in (2, 3, 4):
            if len(script.parents) > depth:
                candidates.append(script.parents[depth] / "lasertracking")
        for c in candidates:
            if c.is_dir() and (
                (c / "serial_comm.py").is_file()
                or (c / "tracking_system_old" / "serial_comm.py").is_file()
            ):
                return c
        return candidates[0] if candidates else Path.cwd() / "lasertracking"

    def _init_hw_serial(self) -> None:
        try:
            lasertracking_dir = self._resolve_lasertracking_dir()
            if str(lasertracking_dir) not in sys.path:
                sys.path.insert(0, str(lasertracking_dir))
            # serial_comm.py 在 tracking_system_old/ 子目录时也加入搜索路径
            old_dir = lasertracking_dir / "tracking_system_old"
            if old_dir.is_dir() and str(old_dir) not in sys.path:
                sys.path.insert(0, str(old_dir))
            from serial_comm import create_serial  # type: ignore

            link = str(self.get_parameter("hw_comm_link").value).strip().lower()
            serial_port = str(self.get_parameter("hw_serial_port").value).strip()
            usb_port = str(self.get_parameter("hw_usb_serial_port").value).strip()
            if link in ("usb", "cdc", "acm"):
                port = usb_port if usb_port else serial_port
                link_label = "usb(cdc)"
            else:
                if link not in ("serial", "uart", "rs232", ""):
                    self.get_logger().warn(
                        f"未知 hw_comm_link={link!r}，按 serial 使用 hw_serial_port"
                    )
                port = serial_port
                link_label = "serial"
            baudrate = int(self.get_parameter("hw_serial_baudrate").value)
            protocol = str(self.get_parameter("hw_serial_protocol").value).strip().lower()
            self._serial = create_serial(
                port=port,
                baudrate=baudrate,
                raw_payload=(protocol != "framed"),
                protocol_mode=protocol,
                enable_feedback=self._hw_serial_enable_feedback,
            )
            self.get_logger().info(
                f"HW模式下位机链路初始化完成: comm_link={link_label}, port={port}, baudrate={baudrate}, "
                f"protocol={protocol}, feedback={'on' if self._hw_serial_enable_feedback else 'off'}, "
                f"use_feedback_state={'on' if self._hw_use_feedback_state else 'off'}"
            )
        except Exception as exc:
            self.get_logger().error(f"HW模式串口初始化失败，回退仅话题发布: {exc}")
            self._serial = None

    @staticmethod
    def _bayer_encoding_to_cv2(enc_l: str) -> Optional[int]:
        m = {
            "bayer_rg8": cv2.COLOR_BayerRG2BGR,
            "bayer_bg8": cv2.COLOR_BayerBG2BGR,
            "bayer_gb8": cv2.COLOR_BayerGB2BGR,
            "bayer_gr8": cv2.COLOR_BayerGR2BGR,
            "bayer_rggb8": cv2.COLOR_BayerRG2BGR,
            "bayer_bggr8": cv2.COLOR_BayerBG2BGR,
            "bayer_gbrg8": cv2.COLOR_BayerGB2BGR,
            "bayer_grbg8": cv2.COLOR_BayerGR2BGR,
            # 部分驱动使用无下划线的 PixelFormat 名（如 BayerRG8）
            "bayerrg8": cv2.COLOR_BayerRG2BGR,
            "bayerbg8": cv2.COLOR_BayerBG2BGR,
            "bayergb8": cv2.COLOR_BayerGB2BGR,
            "bayergr8": cv2.COLOR_BayerGR2BGR,
        }
        return m.get(enc_l)

    def _numpy_payload_to_bgr(
        self, enc: str, w: int, h: int, data: np.ndarray, row_step: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """将 sensor_msgs/Image 原始 payload 解码为 BGR uint8（与海康 Bayer / Gazebo rgb 对齐）。"""
        hi = int(h)
        wi = int(w)
        if hi <= 0 or wi <= 0:
            return None
        enc_l = str(enc).lower()
        rs = int(row_step) if row_step is not None and int(row_step) > 0 else 0

        if enc_l in ("rgb8", "bgr8"):
            ch = 3
        elif enc_l in ("rgba8", "bgra8"):
            ch = 4
        else:
            ch = 0

        if ch > 0:
            row_bytes = rs if rs >= wi * ch else wi * ch
            min_size = hi * row_bytes
            if data.size < min_size:
                return None
            if row_bytes == wi * ch:
                img = data[: hi * wi * ch].reshape((hi, wi, ch))
            else:
                img = np.empty((hi, wi, ch), dtype=np.uint8)
                for r in range(hi):
                    row = data[r * row_bytes : r * row_bytes + wi * ch]
                    img[r, :, :] = row.reshape((wi, ch))
            if enc_l == "rgb8":
                return img[:, :, ::-1].copy()
            if enc_l == "bgr8":
                return np.ascontiguousarray(img)
            if enc_l == "rgba8":
                return img[:, :, :3][:, :, ::-1].copy()
            return np.ascontiguousarray(img[:, :, :3])

        if enc_l in ("mono8", "8uc1"):
            row_bytes = rs if rs >= wi else wi
            if data.size < hi * row_bytes:
                return None
            if row_bytes == wi:
                gray = data[: hi * wi].reshape((hi, wi))
            else:
                gray = np.empty((hi, wi), dtype=np.uint8)
                for r in range(hi):
                    gray[r, :] = data[r * row_bytes : r * row_bytes + wi]
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if enc_l in ("yuy2", "yuyv", "yuv422_yuy2"):
            row_bytes = rs if rs >= wi * 2 else wi * 2
            if data.size < hi * row_bytes:
                return None
            if row_bytes == wi * 2:
                raw = data[: hi * wi * 2].reshape((hi, wi, 2))
            else:
                raw = np.empty((hi, wi, 2), dtype=np.uint8)
                for r in range(hi):
                    sl = data[r * row_bytes : r * row_bytes + wi * 2]
                    raw[r, :, :] = sl.reshape((wi, 2))
            return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUY2)

        if enc_l in ("uyvy", "yuv422"):
            row_bytes = rs if rs >= wi * 2 else wi * 2
            if data.size < hi * row_bytes:
                return None
            if row_bytes == wi * 2:
                raw = data[: hi * wi * 2].reshape((hi, wi, 2))
            else:
                raw = np.empty((hi, wi, 2), dtype=np.uint8)
                for r in range(hi):
                    sl = data[r * row_bytes : r * row_bytes + wi * 2]
                    raw[r, :, :] = sl.reshape((wi, 2))
            return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_UYVY)

        cv_bayer = self._bayer_encoding_to_cv2(enc_l)
        if cv_bayer is not None:
            row_bytes = rs if rs >= wi else wi
            if data.size < hi * row_bytes:
                return None
            if row_bytes == wi:
                raw = data[: hi * wi].reshape((hi, wi))
            else:
                raw = np.empty((hi, wi), dtype=np.uint8)
                for r in range(hi):
                    raw[r, :] = data[r * row_bytes : r * row_bytes + wi]
            return cv2.cvtColor(raw, cv_bayer)

        return None

    def _image_msg_to_bgr(self, msg: Image) -> Optional[np.ndarray]:
        h = int(msg.height)
        w = int(msg.width)
        if h <= 0 or w <= 0:
            return None
        data = np.frombuffer(msg.data, dtype=np.uint8)
        st = int(getattr(msg, "step", 0) or 0)
        out = self._numpy_payload_to_bgr(str(msg.encoding), w, h, data, st if st > 0 else None)
        if out is None:
            enc_s = str(msg.encoding)
            if self._last_image_decode_warn_enc != enc_s:
                self._last_image_decode_warn_enc = enc_s
                self.get_logger().warn(
                    f"暂不支持的图像编码或长度不足: encoding={enc_s} size={data.size} {w}x{h}"
                )
        return out

    def _bgr_from_image_bytes(
        self, enc: str, w: int, h: int, data: bytes
    ) -> Optional[np.ndarray]:
        """与 _image_msg_to_bgr 等价，供后台线程从已拷贝的 payload 解码。"""
        arr = np.frombuffer(data, dtype=np.uint8)
        return self._numpy_payload_to_bgr(enc, w, h, arr)

    def _draw_tele_debug_layers(
        self, dbg: np.ndarray, _w: int, _h: int, ov: TeleDebugOverlay
    ) -> None:
        """调试图 / imshow：仅检测框、中心圆点、辅助重获状态字串。"""
        if ov.have_det and ov.det_xyxy is not None:
            x1, y1, x2, y2 = ov.det_xyxy
            cx_i, cy_i = int(round(ov.det_cx)), int(round(ov.det_cy))
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(dbg, (cx_i, cy_i), 4, (0, 255, 0), -1)
        cv2.putText(
            dbg,
            ov.reacq_mode,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255) if ov.reacq_active else (200, 200, 200),
            2,
        )

    def _draw_wide_debug_layers(self, dbg: np.ndarray, ov: WideDebugOverlay) -> None:
        """短焦调试图：仅检测框与中心圆点。"""
        if ov.have_det and ov.det_xyxy is not None:
            wcx_i, wcy_i = int(round(ov.det_cx)), int(round(ov.det_cy))
            wx1, wy1, wx2, wy2 = ov.det_xyxy
            cv2.rectangle(dbg, (wx1, wy1), (wx2, wy2), (255, 200, 0), 2)
            cv2.circle(dbg, (wcx_i, wcy_i), 4, (255, 200, 0), -1)

    def _shrink_bgr_to_max_width(self, bgr: np.ndarray, max_width: int) -> np.ndarray:
        """调试图发布前按最大宽度等比缩小；max_width<=0 表示不缩放。"""
        if max_width <= 0:
            return bgr
        h, w = bgr.shape[:2]
        if w <= max_width:
            return bgr
        nh = max(1, int(round(h * (max_width / float(w)))))
        return cv2.resize(bgr, (max_width, nh), interpolation=cv2.INTER_AREA)

    def _shrink_bgr_for_tele_debug(self, bgr: np.ndarray) -> np.ndarray:
        """长焦调试图发布前限宽，缩短 tobytes/发布耗时，便于接近 loop_hz（跟踪仍用原图）。"""
        return self._shrink_bgr_to_max_width(bgr, self._tele_debug_image_max_width)

    def _shrink_bgr_for_wide_debug(self, bgr: np.ndarray) -> np.ndarray:
        """短焦调试图发布前限宽，逻辑同长焦；检测仍用原图。"""
        return self._shrink_bgr_to_max_width(bgr, self._wide_debug_image_max_width)

    def _maybe_warn_tele_debug_decode(self, encoding: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_tele_dbg_decode_warn_ns < 5_000_000_000:
            return
        self._last_tele_dbg_decode_warn_ns = now_ns
        self.get_logger().warning(
            f"长焦调试图解码失败(encoding={encoding})，需 rgb8/bgr8/rgba8/bgra8/mono8/BayerRG8 等原始 Image；"
            "若为压缩图请走 image_transport 解压或改相机输出。"
        )

    def _run_tele_debug_compose_job(
        self,
        raw: bytes,
        enc: str,
        w: int,
        h: int,
        stamp_sec: int,
        stamp_nanosec: int,
        frame_id: str,
        ov: TeleDebugOverlay,
    ) -> None:
        dbg = self._bgr_from_image_bytes(enc, w, h, raw)
        if dbg is None:
            self._maybe_warn_tele_debug_decode(enc)
            return
        self._draw_tele_debug_layers(dbg, w, h, ov)
        dbg = self._shrink_bgr_for_tele_debug(dbg)
        try:
            st = RosTime()
            st.sec = int(stamp_sec)
            st.nanosec = int(stamp_nanosec)
            msg = self._bgr_to_image_msg(dbg, st, frame_id)
            if self._debug_img_pub is not None:
                self._debug_img_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warning(f"长焦调试图发布失败: {exc}")

    def _run_wide_debug_compose_job(
        self,
        raw: bytes,
        enc: str,
        w: int,
        h: int,
        stamp_sec: int,
        stamp_nanosec: int,
        frame_id: str,
        ov: WideDebugOverlay,
    ) -> None:
        dbg = self._bgr_from_image_bytes(enc, w, h, raw)
        if dbg is None:
            return
        self._draw_wide_debug_layers(dbg, ov)
        dbg = self._shrink_bgr_for_wide_debug(dbg)
        try:
            st = RosTime()
            st.sec = int(stamp_sec)
            st.nanosec = int(stamp_nanosec)
            msg = self._bgr_to_image_msg(dbg, st, frame_id)
            if self._wide_debug_img_pub is not None:
                self._wide_debug_img_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warning(f"短焦调试图发布失败: {exc}")

    def _submit_tele_debug_compose(self, tele_msg: Image, ov: TeleDebugOverlay) -> None:
        if not self._enable_tele_debug_image or self._debug_img_pub is None:
            return
        if self._debug_pub_executor is None:
            dbg = self._image_msg_to_bgr(tele_msg)
            if dbg is None:
                self._maybe_warn_tele_debug_decode(str(tele_msg.encoding))
                return
            tw, th = int(tele_msg.width), int(tele_msg.height)
            self._draw_tele_debug_layers(dbg, tw, th, ov)
            dbg = self._shrink_bgr_for_tele_debug(dbg)
            self._publish_bgr_debug_sync(
                self._debug_img_pub, dbg, tele_msg.header.stamp, tele_msg.header.frame_id
            )
            return
        prev = self._tele_debug_fut
        if self._async_debug_drop and prev is not None and not prev.done():
            return
        raw = bytes(tele_msg.data)
        enc = str(tele_msg.encoding)
        w, h = int(tele_msg.width), int(tele_msg.height)
        st = tele_msg.header.stamp
        fid = str(tele_msg.header.frame_id)
        node = self
        ov_copy = ov

        def _job() -> None:
            node._run_tele_debug_compose_job(
                raw, enc, w, h, int(st.sec), int(st.nanosec), fid, ov_copy
            )

        self._tele_debug_fut = self._debug_pub_executor.submit(_job)

    def _submit_wide_debug_compose(self, wide_msg: Image, ov: WideDebugOverlay) -> None:
        if not self._enable_wide_debug_image or self._wide_debug_img_pub is None:
            return
        if self._debug_pub_executor is None:
            dbg = self._image_msg_to_bgr(wide_msg)
            if dbg is None:
                return
            self._draw_wide_debug_layers(dbg, ov)
            dbg = self._shrink_bgr_for_wide_debug(dbg)
            self._publish_bgr_debug_sync(
                self._wide_debug_img_pub,
                dbg,
                wide_msg.header.stamp,
                wide_msg.header.frame_id,
            )
            return
        prev = self._wide_debug_fut
        if self._async_debug_drop and prev is not None and not prev.done():
            return
        raw = bytes(wide_msg.data)
        enc = str(wide_msg.encoding)
        w, h = int(wide_msg.width), int(wide_msg.height)
        st = wide_msg.header.stamp
        fid = str(wide_msg.header.frame_id)
        node = self
        ov_copy = ov

        def _job() -> None:
            node._run_wide_debug_compose_job(
                raw, enc, w, h, int(st.sec), int(st.nanosec), fid, ov_copy
            )

        self._wide_debug_fut = self._debug_pub_executor.submit(_job)

    def _log_torch_cuda_vs_infer_devices(self) -> None:
        """YAML 若要求 CUDA，校验 torch.cuda；失败时给出与本机内核匹配的 kmod 安装提示。"""
        try:
            import torch
        except Exception:
            return
        dev_strs: list[str] = []
        for key in ("tele_infer_device", "tele_aerial_infer_device", "wide_infer_device"):
            try:
                dev_strs.append(str(self.get_parameter(key).value))
            except Exception:
                continue
        if not any("cuda" in s.lower() for s in dev_strs):
            return
        ver = getattr(torch, "__version__", "unknown")
        if torch.cuda.is_available():
            try:
                name0 = torch.cuda.get_device_name(0)
            except Exception:
                name0 = "?"
            self.get_logger().info(
                f"PyTorch CUDA 可用: torch={ver}, count={torch.cuda.device_count()}, cuda:0={name0}"
            )
            return
        rel = os.uname().release
        self.get_logger().error(
            "配置为 CUDA 推理（tele/wide*_infer_device），但 torch.cuda.is_available()=False；"
            f"多为未加载 NVIDIA 内核模块（无 /dev/nvidia*），与 pip 是否为 +cpu 无关。当前内核 {rel}，torch={ver}。"
            f"可执行: sudo apt update && sudo apt install -y linux-modules-nvidia-580-{rel} "
            "然后 sudo modprobe nvidia 或重启；脚本: radar_gimbal_gazebo/scripts/setup_nvidia_kernel_modules.sh。"
            "若暂时无 GPU，请把 lasertracking_hw_params.yaml 中 *_infer_device 改为 cpu。"
        )

    def _predict_best_on_frame(
        self,
        frame: np.ndarray,
        model,
        predict_kw: dict,
        min_conf: float,
        min_cx: Optional[float] = None,
    ) -> Optional[Detection]:
        if model is None:
            return None
        try:
            with self._infer_lock:
                results = model.predict(source=frame, **predict_kw)
        except Exception:
            return None
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        best = None
        best_conf = min_conf
        for box in boxes:
            conf = float(box.conf.item())
            if conf < best_conf:
                continue
            xyxy = box.xyxy[0].tolist()
            cx = 0.5 * (xyxy[0] + xyxy[2])
            if min_cx is not None and cx < min_cx:
                continue
            cy = 0.5 * (xyxy[1] + xyxy[3])
            best = Detection(
                cx=cx,
                cy=cy,
                conf=conf,
                xyxy=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
            )
            best_conf = conf
        return best

    def _detect(
        self, msg: Optional[Image], branch: str
    ) -> Tuple[Optional[Detection], Optional[Tuple[int, int]]]:
        if msg is None:
            return None, None
        frame = self._image_msg_to_bgr(msg)
        if frame is None:
            return None, None
        if branch == "wide":
            frame = self._apply_exposure_preprocess(
                frame, self._wide_exposure_gain, self._wide_exposure_offset
            )
        elif self._enable_dynamic_tele_exposure:
            if self._tele_exposure_mode == "search":
                frame = self._apply_exposure_preprocess(
                    frame,
                    self._tele_exposure_search_gain,
                    self._tele_exposure_search_offset,
                )
            else:
                frame = self._apply_exposure_preprocess(
                    frame,
                    self._tele_exposure_track_gain,
                    self._tele_exposure_track_offset,
                )
        h, w = frame.shape[:2]
        if branch == "tele":
            best = self._predict_best_on_frame(
                frame, self._model_tele, self._predict_kwargs_tele, self._min_conf_tele
            )
        else:
            wide_min_cx = 0.5 * w if self._wide_detect_right_half_only else None
            best = self._predict_best_on_frame(
                frame,
                self._model_wide,
                self._predict_kwargs_wide,
                self._min_conf_wide,
                min_cx=wide_min_cx,
            )
        return best, (w, h)

    def _start_detect_workers(self) -> None:
        tele_hz = max(0.5, float(self.get_parameter("tele_detect_hz").value))
        self._tele_worker = threading.Thread(
            target=self._tele_detect_worker,
            args=(tele_hz,),
            name="tele_detect_worker",
            daemon=True,
        )
        self._tele_worker.start()

        if self._enable_wide_detection and self._detector_wide_ready:
            self._wide_worker = threading.Thread(
                target=self._wide_detect_worker,
                args=(),
                name="wide_detect_worker",
                daemon=True,
            )
            self._wide_worker.start()

    def _tele_detect_worker(self, detect_hz: float) -> None:
        period = 1.0 / max(0.5, detect_hz)
        while self._running:
            try:
                tele_msg = self._tele_latest
                if tele_msg is None:
                    time.sleep(period)
                    continue
                frame = self._image_msg_to_bgr(tele_msg)
                if frame is None:
                    time.sleep(period)
                    continue
                if self._enable_dynamic_tele_exposure:
                    if self._tele_exposure_mode == "search":
                        frame = self._apply_exposure_preprocess(
                            frame,
                            self._tele_exposure_search_gain,
                            self._tele_exposure_search_offset,
                        )
                    else:
                        frame = self._apply_exposure_preprocess(
                            frame,
                            self._tele_exposure_track_gain,
                            self._tele_exposure_track_offset,
                        )
                h, w = frame.shape[:2]

                direct_det = self._predict_best_on_frame(
                    frame, self._model_tele, self._predict_kwargs_tele, self._min_conf_tele
                )
                conf_gate = max(self._min_conf_tele, self._tele_direct_conf_gate)
                direct_conf_ok = direct_det is not None and direct_det.conf >= conf_gate
                need_roi_fallback = not direct_conf_ok

                aerial_bbox = None
                roi_det = None
                if need_roi_fallback:
                    aerial_det = self._predict_best_on_frame(
                        frame,
                        self._model_tele_aerial,
                        self._predict_kwargs_tele_aerial,
                        self._min_conf_tele_aerial,
                    )
                    aerial_bbox = aerial_det.xyxy if aerial_det is not None else None
                    if aerial_bbox is not None:
                        x1, y1, x2, y2 = aerial_bbox
                        x1 = max(0, min(w - 1, x1))
                        y1 = max(0, min(h - 1, y1))
                        x2 = max(0, min(w, x2))
                        y2 = max(0, min(h, y2))
                        if x2 - x1 >= 8 and y2 - y1 >= 8:
                            roi = frame[y1:y2, x1:x2]
                            roi_raw = self._predict_best_on_frame(
                                roi, self._model_tele, self._predict_kwargs_tele, self._min_conf_tele
                            )
                            if roi_raw is not None:
                                rx1, ry1, rx2, ry2 = roi_raw.xyxy
                                roi_det = Detection(
                                    cx=roi_raw.cx + x1,
                                    cy=roi_raw.cy + y1,
                                    conf=roi_raw.conf,
                                    xyxy=(rx1 + x1, ry1 + y1, rx2 + x1, ry2 + y1),
                                )

                if direct_conf_ok:
                    det = direct_det
                    det_method = "direct"
                elif roi_det is not None:
                    det = roi_det
                    det_method = "aerial_roi"
                elif direct_det is not None:
                    det = direct_det
                    det_method = "direct_lowconf"
                else:
                    det = None
                    det_method = "none"

                with self._det_lock:
                    self._tele_det_cached = det
                    self._tele_size_cached = (w, h)
                    self._tele_det_method_cached = det_method
                    self._tele_aerial_bbox_cached = aerial_bbox
            except Exception as exc:
                self.get_logger().error(f"长焦检测线程异常: {exc}")
            time.sleep(period)

    def _wide_detect_worker(self) -> None:
        while self._running:
            try:
                wide_msg = self._wide_latest
                det, size = self._detect(wide_msg, "wide")
                with self._det_lock:
                    self._wide_det_cached = det
                    self._wide_size_cached = size
            except Exception as exc:
                self.get_logger().error(f"短焦检测线程异常: {exc}")
            hz = max(0.2, float(self._wide_detect_hz_current))
            period = 1.0 / hz
            time.sleep(period)

    def _bgr_to_image_msg(self, frame: np.ndarray, stamp, frame_id: str) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(frame.shape[0])
        msg.width = int(frame.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = int(frame.shape[1] * frame.shape[2])
        msg.data = frame.tobytes()
        return msg

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        return lo if v < lo else hi if v > hi else v

    def _get_yaw_clamp_range(self) -> tuple[float, float]:
        # 约定: yaw(roll) > 0 表示向左转；左限位用于限制追踪左侧目标。
        yaw_right_lim = abs(float(self.get_parameter("yaw_limit_rad").value))
        yaw_left_lim = abs(float(self.get_parameter("yaw_left_limit_rad").value))
        return -yaw_right_lim, yaw_left_lim

    def _send_gimbal_control(self, yaw_err: float, pitch_err: float, max_step: float) -> None:
        # 弧度误差 -> 度；可选卡尔曼预测补偿延迟
        if self._enable_kalman:
            pred_pitch_deg, pred_roll_deg = self._kf.update(
                math.degrees(pitch_err), math.degrees(yaw_err)
            )
        else:
            pred_pitch_deg = math.degrees(pitch_err)
            pred_roll_deg = math.degrees(yaw_err)
        yaw_abs = abs(pred_roll_deg)
        pitch_abs = abs(pred_pitch_deg)

        yaw_in_zone = yaw_abs < self._dz_yaw
        if self._pitch_deadzone_latched:
            if pitch_abs > (self._dz_pitch * self._dz_pitch_exit_ratio):
                self._pitch_deadzone_latched = False
        else:
            if pitch_abs < self._dz_pitch:
                self._pitch_deadzone_latched = True
        pitch_in_zone = self._pitch_deadzone_latched

        if yaw_in_zone and pitch_in_zone:
            if self._enable_kalman:
                self._kf.zero_velocity()
        else:
            step_yaw = self._clamp(self._kp_yaw * math.radians(pred_roll_deg), -max_step, max_step)
            step_pitch = self._clamp(self._kp_pitch * math.radians(pred_pitch_deg), -max_step, max_step)
            target_yaw = self._yaw + step_yaw
            target_pitch = self._pitch + step_pitch
            self._yaw = self._yaw + self._smooth_yaw * (target_yaw - self._yaw)
            self._pitch = self._pitch + self._smooth_pitch * (target_pitch - self._pitch)

        yaw_min, yaw_max = self._get_yaw_clamp_range()
        pitch_lim = abs(float(self.get_parameter("pitch_limit_rad").value))
        self._yaw = self._clamp(self._yaw, yaw_min, yaw_max)
        self._pitch = self._clamp(self._pitch, -pitch_lim, pitch_lim)



        v_pitch = 0.0
        v_roll = 0.0
        if self._enable_velocity_ff and self._enable_kalman:
            v_pitch_dps, v_roll_dps = self._kf.get_velocity_deg_s()
            yaw_decay_th = max(1e-6, self._dz_yaw * self._velocity_decay_multiple)
            pitch_decay_th = max(1e-6, self._dz_pitch * self._velocity_decay_multiple)
            yaw_decay = min(1.0, max(0.0, yaw_abs / yaw_decay_th)) ** 3
            pitch_decay = min(1.0, max(0.0, pitch_abs / pitch_decay_th)) ** 3
            v_roll = math.radians(v_roll_dps * yaw_decay)
            v_pitch = math.radians(v_pitch_dps * pitch_decay)

        self._emit_pose_command(v_pitch, v_roll)

    def _emit_pose_command(self, v_pitch: float, v_roll: float) -> None:
        """将当前 pitch/yaw 与角速度前馈发到仿真或实机（与重获扫描/短焦辅助共用，避免仅发 ROS 话题而串口无动作）。"""
        msg = Float64MultiArray()
        msg.data = [self._pitch, self._yaw, v_pitch, v_roll]
        if self._runtime_mode == "sim":
            self._upper_pub.publish(msg)
            return

        if self._publish_upper_cmd_in_hw:
            self._upper_pub.publish(msg)
        if self._serial is not None:
            try:
                self._serial.send_pitch_yaw_velocity(
                    math.degrees(self._pitch),
                    math.degrees(self._yaw),
                    math.degrees(v_pitch),
                    math.degrees(v_roll),
                )
                if self._hw_serial_enable_feedback:
                    state = self._serial.poll_feedback()
                    if (
                        self._hw_use_feedback_state
                        and state is not None
                        and (time.time() - float(state.get("recv_time", 0.0))) <= self._hw_feedback_max_age_s
                    ):
                        self._pitch = math.radians(float(state.get("pitch", math.degrees(self._pitch))))
                        self._yaw = math.radians(float(state.get("roll", math.degrees(self._yaw))))
            except Exception as exc:
                self.get_logger().error(f"HW串口发送失败: {exc}")
        else:
            self._upper_pub.publish(msg)

    def _publish_bgr_debug_sync(self, pub, bgr: np.ndarray, stamp, frame_id: str) -> None:
        if pub is None:
            return
        msg = self._bgr_to_image_msg(bgr, stamp, frame_id)
        pub.publish(msg)

    def _schedule_bgr_debug_publish(
        self,
        pub,
        bgr: np.ndarray,
        stamp,
        frame_id: str,
        slot: str,
    ) -> None:
        """slot: 'tele' | 'wide'。异步时在后台线程执行 tobytes + publish。"""
        if pub is None:
            return
        if self._debug_pub_executor is None:
            out = (
                self._shrink_bgr_for_tele_debug(bgr)
                if slot == "tele"
                else self._shrink_bgr_for_wide_debug(bgr)
            )
            self._publish_bgr_debug_sync(pub, out, stamp, frame_id)
            return
        fut_attr = "_tele_debug_fut" if slot == "tele" else "_wide_debug_fut"
        prev: Optional[Future] = getattr(self, fut_attr)
        if self._async_debug_drop and prev is not None and not prev.done():
            return
        bgr_copy = bgr.copy()
        node_ref = self
        slot_name = slot

        def _job() -> None:
            try:
                out = (
                    node_ref._shrink_bgr_for_tele_debug(bgr_copy)
                    if slot_name == "tele"
                    else node_ref._shrink_bgr_for_wide_debug(bgr_copy)
                )
                msg = node_ref._bgr_to_image_msg(out, stamp, frame_id)
                pub.publish(msg)
            except Exception as exc:
                node_ref.get_logger().warning(f"异步调试图发布失败({slot_name}): {exc}")

        setattr(self, fut_attr, self._debug_pub_executor.submit(_job))

    def _timer_wide_and_tele_control(self, dt: float) -> None:
        """短焦检测话题 + 长焦跟踪/重获（与是否发布调试图无关）。"""
        wide_det: Optional[Detection] = None
        wide_size: Optional[Tuple[int, int]] = None
        reacq_mode = "TRACK"

        if self._enable_wide_detection:
            with self._det_lock:
                wide_det = self._wide_det_cached
                wide_size = self._wide_size_cached
            if wide_det is not None:
                self.get_logger().debug(f"短焦检测目标 conf={wide_det.conf:.3f}")
            wide_msg = self._wide_latest
            if wide_msg is not None and wide_size is not None:
                ww, wh = wide_size
                if wide_det is not None:
                    werr_x = (wide_det.cx - 0.5 * ww) / max(1.0, 0.5 * ww)
                    werr_y = (wide_det.cy - 0.5 * wh) / max(1.0, 0.5 * wh)
                    wide_det_msg = Float64MultiArray()
                    wide_det_msg.data = [
                        float(wide_det.cx),
                        float(wide_det.cy),
                        float(werr_x),
                        float(werr_y),
                        float(wide_det.conf),
                    ]
                    if self._wide_det_pub is not None:
                        self._wide_det_pub.publish(wide_det_msg)

        with self._det_lock:
            tele_det = self._tele_det_cached
            tele_size = self._tele_size_cached
        self._update_exposure_state(tele_det is not None)
        err_x = 0.0
        err_y = 0.0

        if tele_det is not None and tele_size is not None:
            self._tele_has_lock_once = True
            self._tele_lost_count = 0
            self._tele_loss_anchor_valid = False
            wts, hts = tele_size
            self._update_effective_aim_pixel_offset(tele_det)
            target_x = 0.5 * wts + float(self._aim_pixel_offset[0])
            target_y = 0.5 * hts + float(self._aim_pixel_offset[1])
            err_x = (tele_det.cx - target_x) / max(1.0, 0.5 * wts)
            err_y = (tele_det.cy - target_y) / max(1.0, 0.5 * hts)
            yaw_sign = float(self.get_parameter("yaw_error_sign").value)
            pitch_sign = float(self.get_parameter("pitch_error_sign").value)
            # 针孔相机精确公式: θ = atan(norm_offset * tan(hfov/2))
            yaw_err = yaw_sign * math.atan(err_x * math.tan(self._tele_hfov * 0.5))
            pitch_err = pitch_sign * math.atan((-err_y) * math.tan(self._tele_vfov * 0.5))
            max_step = max(0.001, float(self.get_parameter("max_step_rad").value))
            self._send_gimbal_control(yaw_err, pitch_err, max_step)
            if self._reacq_active:
                self._reacq_seen_count += 1
                if self._reacq_seen_count >= self._reacq_confirm_frames:
                    self._stop_reacq("tele confirmed")
                else:
                    reacq_mode = "REACQ_CONFIRM"
            else:
                self._reacq_seen_count = 0
        else:
            if self._reacq_require_first_lock and not self._tele_has_lock_once:
                self._tele_lost_count = 0
                reacq_mode = "WAIT_FIRST_LOCK"
            else:
                if self._tele_has_lock_once and self._tele_lost_count == 0:
                    self._tele_loss_anchor_yaw = float(self._yaw)
                    self._tele_loss_anchor_pitch = float(self._pitch)
                    self._tele_loss_anchor_valid = True
                self._tele_lost_count += 1
            self._reacq_seen_count = 0
            if self._tele_has_lock_once or (not self._reacq_require_first_lock):
                if (
                    self._enable_reacq_assist
                    and (self._enable_reacq_scan or self._enable_reacq_wide_assist)
                    and self._tele_lost_count >= self._reacq_enter_lost_frames
                ):
                    if not self._reacq_active:
                        self._start_reacq()
                    reacq_mode = self._run_reacq_assist(wide_det, wide_size, dt)

        self._debug_overlay_err_x = err_x
        self._debug_overlay_err_y = err_y
        self._debug_overlay_reacq_mode = reacq_mode

    def _debug_publish_async_overlays(self, tele_msg: Image) -> None:
        """主线程只做快照；长/短焦解码+绘制+发布在线程池。先提交长焦，减轻 async 丢帧时长焦饿死。"""
        with self._det_lock:
            tele_det = self._tele_det_cached
            tele_det_method = self._tele_det_method_cached
            tele_aerial_bbox = self._tele_aerial_bbox_cached
        d_text = (
            "--"
            if self._parallax_last_distance_m is None
            else f"{self._parallax_last_distance_m:.2f}m"
        )
        tov = TeleDebugOverlay(
            have_det=tele_det is not None,
            det_xyxy=tele_det.xyxy if tele_det is not None else None,
            det_cx=tele_det.cx if tele_det is not None else 0.0,
            det_cy=tele_det.cy if tele_det is not None else 0.0,
            det_conf=tele_det.conf if tele_det is not None else 0.0,
            det_method=tele_det_method,
            aerial_bbox=tele_aerial_bbox,
            err_x=self._debug_overlay_err_x,
            err_y=self._debug_overlay_err_y,
            reacq_mode=self._debug_overlay_reacq_mode,
            tele_lost_count=self._tele_lost_count,
            aim_dx=float(self._aim_pixel_offset[0]),
            aim_dy=float(self._aim_pixel_offset[1]),
            boresight_src=self._boresight_source,
            yaw=self._yaw,
            pitch=self._pitch,
            reacq_active=self._reacq_active,
            parallax_on=self._parallax_enabled,
            parallax_ox=self._parallax_last_offset[0],
            parallax_oy=self._parallax_last_offset[1],
            parallax_range_str=d_text,
            parallax_src=self._parallax_last_distance_source,
            dyn_exp_on=self._enable_dynamic_tele_exposure,
            tele_exp_mode=self._tele_exposure_mode,
            det_streak=self._tele_detect_streak,
            lost_streak=self._tele_lost_streak,
        )
        self._submit_tele_debug_compose(tele_msg, tov)

        if self._enable_wide_debug_image and self._enable_wide_detection:
            with self._det_lock:
                wide_det = self._wide_det_cached
                wide_size = self._wide_size_cached
            wide_msg = self._wide_latest
            if wide_msg is not None and wide_size is not None:
                wov = WideDebugOverlay(
                    have_det=wide_det is not None,
                    det_xyxy=wide_det.xyxy if wide_det is not None else None,
                    det_cx=wide_det.cx if wide_det is not None else 0.0,
                    det_cy=wide_det.cy if wide_det is not None else 0.0,
                    det_conf=wide_det.conf if wide_det is not None else 0.0,
                )
                self._submit_wide_debug_compose(wide_msg, wov)

    def _on_timer_sync_debug(self, tele_msg: Image) -> None:
        """同步路径：仅解码/绘制/发布调试图（控制已在 _timer_wide_and_tele_control 中执行）。"""
        if self._enable_wide_debug_image and self._enable_wide_detection:
            with self._det_lock:
                wide_det = self._wide_det_cached
            if wide_det is not None:
                self.get_logger().debug(f"短焦检测目标 conf={wide_det.conf:.3f}")
            wide_msg = self._wide_latest
            wide_dbg = self._image_msg_to_bgr(wide_msg) if wide_msg is not None else None
            if wide_dbg is not None and wide_msg is not None:
                if wide_det is not None:
                    wcx_i, wcy_i = int(round(wide_det.cx)), int(round(wide_det.cy))
                    wx1, wy1, wx2, wy2 = wide_det.xyxy
                    cv2.rectangle(wide_dbg, (wx1, wy1), (wx2, wy2), (255, 200, 0), 2)
                    cv2.circle(wide_dbg, (wcx_i, wcy_i), 4, (255, 200, 0), -1)
                self._schedule_bgr_debug_publish(
                    self._wide_debug_img_pub,
                    wide_dbg,
                    wide_msg.header.stamp,
                    wide_msg.header.frame_id,
                    "wide",
                )

        if not self._enable_tele_debug_image:
            return

        dbg = self._image_msg_to_bgr(tele_msg)
        if dbg is None:
            self._maybe_warn_tele_debug_decode(str(tele_msg.encoding))
            return

        with self._det_lock:
            tele_det = self._tele_det_cached
            tele_size = self._tele_size_cached

        reacq_mode = self._debug_overlay_reacq_mode

        if tele_det is not None and tele_size is not None:
            cx_i, cy_i = int(round(tele_det.cx)), int(round(tele_det.cy))
            x1, y1, x2, y2 = tele_det.xyxy
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(dbg, (cx_i, cy_i), 4, (0, 255, 0), -1)

        cv2.putText(
            dbg,
            reacq_mode,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255) if self._reacq_active else (200, 200, 200),
            2,
        )
        self._schedule_bgr_debug_publish(
            self._debug_img_pub,
            dbg,
            tele_msg.header.stamp,
            tele_msg.header.frame_id,
            "tele",
        )

    def _on_timer(self) -> None:
        try:
            now_ts = time.time()
            dt = max(1e-3, min(0.10, now_ts - self._last_timer_ts))
            self._last_timer_ts = now_ts
            if not self._detector_ready or self._model_tele is None:
                now_ns = self.get_clock().now().nanoseconds
                if now_ns - self._last_missing_dep_warn_ns > 5_000_000_000:
                    self.get_logger().warn(
                        "检测器未就绪（缺少 ultralytics 或模型加载失败），等待依赖安装后重启节点。"
                    )
                    self._last_missing_dep_warn_ns = now_ns
                return

            tele_msg = self._tele_latest
            if tele_msg is None:
                return
            tw = int(tele_msg.width)
            th = int(tele_msg.height)
            if tw <= 0 or th <= 0:
                return

            self._timer_wide_and_tele_control(dt)

            if not (self._enable_tele_debug_image or self._enable_wide_debug_image):
                return

            if self._async_debug_publish and self._debug_pub_executor is not None:
                self._debug_publish_async_overlays(tele_msg)
            else:
                self._on_timer_sync_debug(tele_msg)
        except Exception as exc:
            now_ns = self.get_clock().now().nanoseconds
            # 限流错误日志，避免刷屏
            if now_ns - self._last_runtime_error_ns > 1_000_000_000:
                self.get_logger().error(f"运行时异常（已拦截，节点不中断）: {exc}")
                self._last_runtime_error_ns = now_ns

    def destroy_node(self):
        self._stop_tele_raw_recording()
        self._stop_tele_game_folder_images()
        self._running = False
        if self._tele_worker is not None and self._tele_worker.is_alive():
            self._tele_worker.join(timeout=0.5)
        if self._wide_worker is not None and self._wide_worker.is_alive():
            self._wide_worker.join(timeout=0.5)
        if self._debug_pub_executor is not None:
            try:
                self._debug_pub_executor.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass
            self._debug_pub_executor = None
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = LaserTrackingTracker()
    try:
        rclpy.spin(node)
    finally:
        if node._serial is not None:
            try:
                node._serial.close()
            except Exception:
                pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
