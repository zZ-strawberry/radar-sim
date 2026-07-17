"""激光-相机 boresight 交互式校准工具。

模式 A（有模型）: python ... --config config_tracking.yaml
模式 B（手动）:   python ... --config config_tracking.yaml --no-detector
"""


from __future__ import annotations

import argparse
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
import importlib
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))        # .../lasertracking/calibration
LASERTRACKING_DIR = os.path.dirname(CURRENT_DIR)                 # .../lasertracking
PROJECT_ROOT = os.path.dirname(LASERTRACKING_DIR)                # .../PFA_radar-2026-main

for p in (CURRENT_DIR, LASERTRACKING_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.append(p)


class ThreadedCamera:
    """线程化相机包装器 - 独立线程持续采集最新帧。"""

    def __init__(self, camera, enable_thread=True):
        self.camera = camera
        self.enable_thread = bool(enable_thread)
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.stopped = True
        self.frame_count = 0
        self.thread = None
        if self.enable_thread:
            self._start_thread(wait_first_frame=True)

    def _start_thread(self, wait_first_frame=False):
        if self.thread is not None and self.thread.is_alive():
            return
        self.stopped = False
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        if wait_first_frame:
            deadline = time.time() + 5.0
            while time.time() < deadline:
                with self.lock:
                    if self.frame is not None:
                        break
                time.sleep(0.05)

    def _stop_thread(self):
        if self.thread is None:
            return
        self.stopped = True
        self.thread.join(timeout=2.0)
        self.thread = None

    def set_thread_mode(self, enable_thread):
        enable_thread = bool(enable_thread)
        if enable_thread == self.enable_thread:
            return
        self.enable_thread = enable_thread
        if self.enable_thread:
            self._start_thread(wait_first_frame=False)
        else:
            self._stop_thread()

    def _update_loop(self):
        while not self.stopped:
            try:
                ret, frame = self.camera.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                        self.ret = True
                        self.frame_count += 1
            except Exception:
                time.sleep(0.01)

    def read(self):
        if not self.enable_thread:
            return self.camera.read()
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        if self.enable_thread:
            self._stop_thread()
        self.camera.release()

    def set_camera_params(self, exposure_time=None, gain=None):
        return self.camera.set_camera_params(exposure_time, gain)

    def get_camera_params(self):
        return self.camera.get_camera_params()


def parallax_offset(fx, fy, bx, by, z, z_ref=None):
    """返回距离 z(m) 处激光光斑相对瞄准点的像素偏置 (du, dv)。
    若给定 z_ref>0，返回以 z_ref 处为 0 参考的相对偏置。
    """
    if z <= 1e-9:
        return 0.0, 0.0
    du = fx * bx / z
    dv = fy * by / z
    if z_ref is not None and z_ref > 0.0:
        du -= fx * bx / z_ref
        dv -= fy * by / z_ref
    return du, dv


WINDOW_NAME = "boresight_calibrator"

# 方向键在不同 OpenCV 版本 / 平台下的原始键码
ARROW_LEFT = (2424832, 65361, 81)
ARROW_RIGHT = (2555904, 65363, 83)
ARROW_UP = (2490368, 65362, 82)
ARROW_DOWN = (2621440, 65364, 84)

# 由 main() 在导入海康 SDK 前填入：MvImport 包所在目录的**父目录**（其父下应有子目录 MvImport/）
_EXTRA_MVIMPORT_ROOTS: List[str] = []


def _collect_mvimport_roots() -> List[str]:
    roots: List[str] = []
    for r in _EXTRA_MVIMPORT_ROOTS:
        if r and os.path.isdir(r):
            roots.append(os.path.abspath(r))
    envp = os.environ.get("MVIMPORT_PATH", "").strip()
    if envp and os.path.isdir(envp):
        roots.append(os.path.abspath(envp))
    for rel in (
        "",
        "MvImport_Linux",
        os.path.join("MvCamCtrlSDK_STD_V4.7.0_251113", "Samples", "Python"),
    ):
        cand = os.path.join(PROJECT_ROOT, rel) if rel else PROJECT_ROOT
        if os.path.isdir(cand):
            roots.append(os.path.abspath(cand))
    out: List[str] = []
    seen = set()
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _load_hik_sdk():
    last_err: Optional[BaseException] = None
    for root in _collect_mvimport_roots():
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            return importlib.import_module("MvImport.MvCameraControl_class")
        except ImportError as exc:
            last_err = exc
            continue
    hint = (
        "请任选其一：\n"
        "  1) 设置环境变量 MVIMPORT_PATH 为包含子目录 MvImport 的父路径；或\n"
        "  2) 运行: python3 .../boresight_calibrator.py --mvimport-path /path/to/parent_of_MvImport；或\n"
        "  3) 先启动海康 ROS2 驱动，再使用: --ros-image-topic /gimbal_camera/image_raw"
    )
    raise RuntimeError(
        "无法导入 MvImport.MvCameraControl_class（海康 Python SDK）。\n" + hint
    ) from last_err


def _ros_image_msg_to_bgr(msg) -> Optional[np.ndarray]:
    """sensor_msgs/Image -> BGR ndarray（与 lasertracking_tracker 逻辑一致）。"""
    h = int(msg.height)
    w = int(msg.width)
    if h <= 0 or w <= 0:
        return None
    enc = str(msg.encoding).lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        ch = 3
    elif enc in ("rgba8", "bgra8"):
        ch = 4
    else:
        return None
    need = h * w * ch
    if data.size < need:
        return None
    img = data[:need].reshape((h, w, ch))
    if enc == "rgb8":
        return img[:, :, ::-1].copy()
    if enc == "bgr8":
        return img
    if enc == "rgba8":
        return img[:, :, :3][:, :, ::-1].copy()
    return img[:, :, :3]


class RosImageCamera:
    """通过 ROS2 订阅图像，无需本机 MvImport（需已 source ROS 且话题有发布）。"""

    def __init__(
        self,
        topic: str,
        exposure_time: float = 0.0,
        gain: float = 0.0,
    ) -> None:
        self.topic = str(topic).strip()
        self.exposure_time = float(exposure_time)
        self.gain = float(gain)
        self._node = None
        self._sub = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._spin_thread: Optional[threading.Thread] = None
        self._running = False
        self._did_rclpy_init = False

    def open(self) -> bool:
        try:
            import rclpy
            from sensor_msgs.msg import Image
        except ImportError as exc:
            print(
                f"ROS2 Python 不可用: {exc}\n"
                "  请先 source 本机发行版，例如: source /opt/ros/humble/setup.bash"
                "（将 humble 换成 jazzy/iron 等实际目录名，勿使用字面量 <distro>）"
            )
            return False
        if not self.topic:
            print("ros-image-topic 为空")
            return False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._did_rclpy_init = True
        from rclpy.qos import qos_profile_sensor_data

        self._node = rclpy.create_node("boresight_calibrator_cam_sub")
        # 与 lasertracking_tracker 一致：相机常用 SensorData/BestEffort，默认 Reliable 会收不到图
        self._sub = self._node.create_subscription(
            Image, self.topic, self._image_cb, qos_profile_sensor_data,
        )
        self._running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()
        t_end = time.time() + 8.0
        while time.time() < t_end:
            with self._lock:
                if self._frame is not None:
                    print(f"ROS 图像已连接: {self.topic}")
                    return True
            time.sleep(0.05)
        print(
            f"超时未收到图像: {self.topic}\n"
            "  请逐项检查:\n"
            "  1) 已 source ROS（勿复制占位符）: source /opt/ros/<你的发行版>/setup.bash\n"
            "  2) 海康/相机 ROS 节点已启动，且正在发布该话题:\n"
            f"       ros2 topic hz {self.topic}\n"
            "  3) 话题名与 lasertracking_hw_params.yaml 里 tele_image_topic 一致"
        )
        self.release()
        return False

    def _image_cb(self, msg) -> None:
        bgr = _ros_image_msg_to_bgr(msg)
        if bgr is not None:
            with self._lock:
                self._frame = bgr

    def _spin_loop(self) -> None:
        import rclpy

        while self._running and rclpy.ok() and self._node is not None:
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def set_camera_params(self, exposure_time=None, gain=None) -> bool:
        print("ROS 模式：曝光/增益请在相机驱动（如 hik_*.yaml 或动态参数）中调整，此处忽略。")
        return True

    def get_camera_params(self) -> Optional[Dict]:
        return None

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self) -> None:
        self._running = False
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self._spin_thread = None
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        self._sub = None
        if self._did_rclpy_init:
            try:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
            self._did_rclpy_init = False
        print("ROS 相机订阅已释放")


def _resolve_path(path_value, *, base_dir=LASERTRACKING_DIR, root_dir=PROJECT_ROOT):
    if path_value is None:
        return None
    path_str = str(path_value)
    if os.path.isabs(path_str):
        return path_str
    for base in (base_dir, root_dir):
        candidate = os.path.join(base, path_str)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(root_dir, path_str)


@dataclass
class CalibratorSettings:
    config_path: str
    output_path: str
    detector_enabled: bool = True
    sync_config: bool = True
    sync_ros2_config: bool = True
    ros2_config_paths: Optional[List[str]] = None
    auto_save_on_exit: bool = True
    step_px: int = 2
    window_size: Tuple[int, int] = (1280, 720)


class MouseState:
    """线程安全的鼠标事件缓冲，主循环里消费。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.event = 0
        self.x = 0
        self.y = 0
        self.flags = 0
        self.view_w = 0
        self.view_h = 0
        self.pending = False


class HikCameraCapture:
    def __init__(self, camera_index=0, exposure_time=55915.0, gain=10.5):
        self.camera_index = int(camera_index)
        self.exposure_time = exposure_time
        self.gain = gain
        self.sdk = None
        self.cam = None
        self.data_buf = None
        self.nPayloadSize = None

    def open(self):
        self.sdk = _load_hik_sdk()
        device_list = self.sdk.MV_CC_DEVICE_INFO_LIST()
        tlayer_type = self.sdk.MV_GIGE_DEVICE | self.sdk.MV_USB_DEVICE
        ret = self.sdk.MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != 0:
            print(f"枚举设备失败! ret[0x{ret:x}]")
            return False
        if device_list.nDeviceNum == 0:
            print("未找到设备!")
            return False
        if self.camera_index < 0 or self.camera_index >= device_list.nDeviceNum:
            print(f"相机索引越界: index={self.camera_index}, devices={device_list.nDeviceNum}")
            return False
        print(f"找到 {device_list.nDeviceNum} 个设备")
        self.cam = self.sdk.MvCamera()
        st_device = cast(
            device_list.pDeviceInfo[self.camera_index],
            POINTER(self.sdk.MV_CC_DEVICE_INFO),
        ).contents
        ret = self.cam.MV_CC_CreateHandle(st_device)
        if ret != 0:
            print(f"创建句柄失败! ret[0x{ret:x}]")
            return False
        ret = self.cam.MV_CC_OpenDevice(self.sdk.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print(f"打开设备失败! ret[0x{ret:x}]")
            self.cam.MV_CC_DestroyHandle()
            self.cam = None
            return False
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", self.sdk.MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print(f"设置触发模式失败! ret[0x{ret:x}]")
        self.set_camera_params(self.exposure_time, self.gain)
        st_param = self.sdk.MVCC_INTVALUE()
        memset(byref(st_param), 0, sizeof(self.sdk.MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", st_param)
        if ret != 0:
            print(f"获取PayloadSize失败! ret[0x{ret:x}]")
            self.release()
            return False
        self.nPayloadSize = int(st_param.nCurValue)
        self.data_buf = (c_ubyte * self.nPayloadSize)()
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"开始取流失败! ret[0x{ret:x}]")
            self.release()
            return False
        print("相机打开成功!")
        return True

    def set_camera_params(self, exposure_time=None, gain=None):
        if self.cam is None:
            print("相机未初始化!")
            return False
        if exposure_time is not None:
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time))
            if ret == 0:
                self.exposure_time = exposure_time
                print(f"✓ 设置曝光时间: {exposure_time} μs")
            else:
                print(f"⚠️ 设置曝光时间失败: 0x{ret:x}")
        if gain is not None:
            ret = self.cam.MV_CC_SetFloatValue("Gain", float(gain))
            if ret == 0:
                self.gain = gain
                print(f"✓ 设置增益: {gain} dB")
            else:
                print(f"⚠️ 设置增益失败: 0x{ret:x}")
        return True

    def get_camera_params(self):
        if self.cam is None:
            print("相机未初始化!")
            return None
        params = {}
        st_float = self.sdk.MVCC_FLOATVALUE()
        memset(byref(st_float), 0, sizeof(self.sdk.MVCC_FLOATVALUE))
        ret = self.cam.MV_CC_GetFloatValue("ExposureTime", st_float)
        if ret == 0:
            params["exposure_time"] = st_float.fCurValue
            params["exposure_min"] = st_float.fMin
            params["exposure_max"] = st_float.fMax
        memset(byref(st_float), 0, sizeof(self.sdk.MVCC_FLOATVALUE))
        ret = self.cam.MV_CC_GetFloatValue("Gain", st_float)
        if ret == 0:
            params["gain"] = st_float.fCurValue
            params["gain_min"] = st_float.fMin
            params["gain_max"] = st_float.fMax
        return params

    def read(self):
        if self.cam is None or self.data_buf is None or self.nPayloadSize is None:
            return False, None
        st_frame_info = self.sdk.MV_FRAME_OUT_INFO_EX()
        memset(byref(st_frame_info), 0, sizeof(st_frame_info))
        ret = self.cam.MV_CC_GetOneFrameTimeout(
            byref(self.data_buf),
            self.nPayloadSize,
            st_frame_info,
            1000,
        )
        if ret != 0:
            return False, None
        frame = self.image_control(self.data_buf, st_frame_info)
        if frame is None:
            return False, None
        return True, frame

    def image_control(self, raw_buf, st_frame_info):
        width = int(st_frame_info.nWidth)
        height = int(st_frame_info.nHeight)
        if st_frame_info.enPixelType == self.sdk.PixelType_Gvsp_BGR8_Packed:
            size_bytes = width * height * 3
            img = np.frombuffer(raw_buf, count=size_bytes, dtype=np.uint8)
            return img.reshape(height, width, 3)
        convert_param = self.sdk.MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(convert_param), 0, sizeof(self.sdk.MV_CC_PIXEL_CONVERT_PARAM))
        convert_param.nWidth = width
        convert_param.nHeight = height
        convert_param.pSrcData = cast(raw_buf, POINTER(c_ubyte))
        convert_param.nSrcDataLen = st_frame_info.nFrameLen
        convert_param.enSrcPixelType = st_frame_info.enPixelType
        convert_param.enDstPixelType = self.sdk.PixelType_Gvsp_BGR8_Packed
        dst_size = width * height * 3
        dst_buf = (c_ubyte * dst_size)()
        convert_param.pDstBuffer = cast(dst_buf, POINTER(c_ubyte))
        convert_param.nDstBufferSize = dst_size
        ret = self.cam.MV_CC_ConvertPixelType(convert_param)
        if ret != 0:
            print(f"像素格式转换失败 ret=0x{ret:x}")
            return None
        return np.frombuffer(dst_buf, count=dst_size, dtype=np.uint8).reshape(height, width, 3)

    def release(self):
        if self.cam is None:
            return
        try:
            self.cam.MV_CC_StopGrabbing()
        except Exception:
            pass
        try:
            self.cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            self.cam.MV_CC_DestroyHandle()
        except Exception:
            pass
        self.cam = None
        print("相机已释放")


def mouse_callback(event, x, y, flags, userdata):
    state: MouseState = userdata
    if state is None:
        return
    with state.lock:
        state.event = event
        state.x = x
        state.y = y
        state.flags = flags
        state.pending = True


def draw_crosshair(img: np.ndarray, p: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    base = max(1, min(img.shape[1], img.shape[0]))
    size = max(16, base // 60)
    thickness = max(1, base // 700)
    halo = thickness + 2
    px, py = int(p[0]), int(p[1])
    cv2.line(img, (px - size, py), (px + size, py), (0, 0, 0), halo, cv2.LINE_AA)
    cv2.line(img, (px, py - size), (px, py + size), (0, 0, 0), halo, cv2.LINE_AA)
    cv2.line(img, (px - size, py), (px + size, py), color, thickness, cv2.LINE_AA)
    cv2.line(img, (px, py - size), (px, py + size), color, thickness, cv2.LINE_AA)


def draw_text_lines(img: np.ndarray, lines: List[str]) -> None:
    x = 30
    y = 60
    base = max(1, min(img.shape[1], img.shape[0]))
    scale = max(0.9, min(base / 900.0, 2.2))
    thickness = max(2, int(round(scale * 2.0)))
    line_step = int(round(32.0 * scale))
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 0, 0), thickness + 1, cv2.LINE_AA)
        y += line_step


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_camera(
    config: Dict,
    ros_image_topic: Optional[str] = None,
) -> ThreadedCamera:
    cam_cfg = config.get("camera", {}) or {}
    camera_index = int(cam_cfg.get("camera_index", 0) or 0)
    exposure = float(cam_cfg.get("exposure_time", 55915.0))
    gain = float(cam_cfg.get("gain", 10.5))
    use_thread = bool(cam_cfg.get("use_thread", True))

    topic = (ros_image_topic or "").strip()
    if topic:
        base_camera = RosImageCamera(
            topic, exposure_time=exposure, gain=gain,
        )
        if not base_camera.open():
            raise RuntimeError("ROS 图像相机打开失败（见上方日志）")
        # RosImageCamera 内部已有订阅与 spin 线程，避免再套一层采集线程
        return ThreadedCamera(base_camera, enable_thread=False)

    base_camera = HikCameraCapture(
        camera_index=camera_index, exposure_time=exposure, gain=gain,
    )
    if not base_camera.open():
        raise RuntimeError("海康相机打开失败，请检查设备连接/驱动/占用情况")
    return ThreadedCamera(base_camera, enable_thread=use_thread)


def build_detector(config: Dict):
    models = config.get("models", {}) or {}
    weights = _resolve_path(models.get("armor", "best.pt"))
    data_yaml = _resolve_path(models.get("data_yaml", "data2_laser.yaml"))
    backend = str(models.get("armor_backend", "yolo26")).lower()

    detector_cfg = (config.get("detector", {}) or {}).get("armor", {}) or {}
    img_size = detector_cfg.get("img_size", [640, 640])
    if not isinstance(img_size, (list, tuple)) or len(img_size) < 2:
        img_size = [640, 640]

    use_gpu = bool((config.get("gpu", {}) or {}).get("enable", False))
    try:
        import torch  # noqa: WPS433
        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    except Exception:
        device = "cpu"

    # AerialDetectorAdapter 需配合旧版 aerial_tracking_system.py 使用，当前不可用。
    # 标定请使用 --no-detector 手动模式，或恢复旧版依赖后重新启用。
    print("检测器不可用（缺少 aerial_tracking_system 依赖），请使用 --no-detector 手动模式")
    return None


def detect_best_target(detector,
                       frame: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int],
                                                            Tuple[float, float],
                                                            float]]:
    """返回 (bbox_xyxy, center, conf) 或 None。选取最高置信度。"""
    try:
        detections = detector.predict(frame)
    except Exception:
        return None
    best = None
    for _cls, xywh, conf in detections:
        if best is None or float(conf) > best[2]:
            ax, ay, aw, ah = xywh
            x1 = int(ax)
            y1 = int(ay)
            x2 = x1 + int(aw)
            y2 = y1 + int(ah)
            center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
            best = ((x1, y1, x2, y2), center, float(conf))
    return best


def read_baseline(config: Dict) -> Tuple[float, float, float]:
    calib = config.get("calibration", {}) or {}
    baseline = calib.get("baseline", {}) or {}
    bx = float(baseline.get("bx", 0.0) or 0.0)
    by = float(baseline.get("by", 0.0) or 0.0)
    z_ref = float(baseline.get("z_ref", 0.0) or 0.0)
    return bx, by, z_ref


def read_intrinsics(config: Dict) -> Tuple[float, float]:
    calib = config.get("calibration", {}) or {}
    fx = float(calib.get("fx", 0.0) or 0.0)
    fy = float(calib.get("fy", 0.0) or 0.0)
    return fx, fy


def print_config_check(config: Dict, settings: CalibratorSettings) -> int:
    print("=== boresight calibrator config check ===")
    print(f"config: {settings.config_path} {'OK' if os.path.exists(settings.config_path) else 'MISSING'}")
    print(f"output: {settings.output_path}")
    if settings.sync_ros2_config and settings.ros2_config_paths:
        for cfg_path in settings.ros2_config_paths:
            print(f"ros2_param_config: {cfg_path} {'OK' if os.path.exists(cfg_path) else 'MISSING'}")
    output_dir = os.path.dirname(os.path.abspath(settings.output_path))
    print(f"output_dir: {output_dir} {'OK' if os.path.isdir(output_dir) else 'WILL_CREATE'}")
    calib = config.get("calibration", {}) or {}
    boresight_cfg = calib.get("boresight_file")
    if boresight_cfg:
        boresight_path = _resolve_path(boresight_cfg)
        print(f"boresight_file: {boresight_path} {'EXISTS' if boresight_path and os.path.exists(boresight_path) else 'NOT_EXISTS'}")
    fx, fy = read_intrinsics(config)
    bx, by, z_ref = read_baseline(config)
    print(f"intrinsics: fx={fx:g} fy={fy:g} {'OK' if fx > 0.0 and fy > 0.0 else 'INVALID'}")
    print(f"baseline: bx={bx:g} by={by:g} z_ref={z_ref:g}")
    cam_cfg = config.get("camera", {}) or {}
    print(
        "camera: "
        f"index={int(cam_cfg.get('camera_index', 0) or 0)} "
        f"exposure={float(cam_cfg.get('exposure_time', 55915.0) or 55915.0):g} "
        f"gain={float(cam_cfg.get('gain', 10.5) or 10.5):g} "
        f"use_thread={bool(cam_cfg.get('use_thread', True))}"
    )
    if settings.detector_enabled:
        models = config.get("models", {}) or {}
        weights = _resolve_path(models.get("armor", "best.pt"))
        data_yaml = _resolve_path(models.get("data_yaml", "data2_laser.yaml"))
        print(f"detector_weights: {weights} {'EXISTS' if weights and os.path.exists(weights) else 'MISSING'}")
        print(f"detector_data_yaml: {data_yaml} {'EXISTS' if data_yaml and os.path.exists(data_yaml) else 'MISSING'}")
    else:
        print("detector: disabled by --no-detector")
    print("check finished")
    return 0


def save_boresight(path: str,
                   u_L: float, v_L: float,
                   image_w: int, image_h: int,
                   bx: float, by: float, z_ref: float,
                   notes: Optional[str] = None) -> bool:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    except OSError:
        pass
    data = {
        "u_L": float(u_L),
        "v_L": float(v_L),
        "image_w": int(image_w),
        "image_h": int(image_h),
    }
    if z_ref > 0.0:
        data["z_ref"] = float(z_ref)
    if abs(bx) > 1e-9 or abs(by) > 1e-9:
        data["bx"] = float(bx)
        data["by"] = float(by)
    data["notes"] = notes or time.strftime("%Y-%m-%d %H:%M:%S 现场校准")
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        return True
    except OSError as exc:
        print(f"[保存] 写 boresight.yaml 失败: {exc}")
        return False


def sync_pixel_offset_to_config(config_path: str, dx: float, dy: float) -> bool:
    """以"最小扰动"的方式把 pixel_offset 写回原 yaml，保留原有注释。"""
    if not config_path or not os.path.exists(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as exc:
        print(f"[同步] 读 config 失败: {exc}")
        return False

    replacement_value = f"[{dx:.2f}, {dy:.2f}]"
    new_lines: List[str] = []
    replaced = False
    for line in lines:
        stripped = line.lstrip()
        if not replaced and stripped.startswith("pixel_offset:"):
            indent = line[: len(line) - len(stripped)]
            comment_idx = line.find("#")
            comment = line[comment_idx:].rstrip("\n") if comment_idx >= 0 else ""
            new_line = f"{indent}pixel_offset: {replacement_value}"
            if comment:
                new_line += f"    {comment}"
            new_line += "\n"
            new_lines.append(new_line)
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        new_lines.append("\n")
        new_lines.append("# pixel_offset auto-synced from boresight_calibrator\n")
        new_lines.append(f"pixel_offset: {replacement_value}\n")

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return True
    except OSError as exc:
        print(f"[同步] 写 config 失败: {exc}")
        return False


def _path_for_yaml(path: str) -> str:
    try:
        abs_path = os.path.abspath(path)
        rel_path = os.path.relpath(abs_path, PROJECT_ROOT)
        if not rel_path.startswith("..") and not os.path.isabs(rel_path):
            return rel_path.replace(os.sep, "/")
    except ValueError:
        pass
    return os.path.abspath(path)


def _replace_scalar_yaml_key(lines: List[str], key: str, value: str) -> Tuple[List[str], bool]:
    new_lines: List[str] = []
    replaced = False
    for line in lines:
        stripped = line.lstrip()
        if not replaced and stripped.startswith(f"{key}:"):
            indent = line[: len(line) - len(stripped)]
            comment_idx = line.find("#")
            comment = line[comment_idx:].rstrip("\n") if comment_idx >= 0 else ""
            new_line = f"{indent}{key}: {value}"
            if comment:
                new_line += f"  {comment}"
            new_line += "\n"
            new_lines.append(new_line)
            replaced = True
        else:
            new_lines.append(line)
    return new_lines, replaced


def _insert_ros2_param(lines: List[str], key: str, value: str) -> List[str]:
    insert_at = len(lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if line.startswith("    ") and stripped and not stripped.startswith("#"):
            insert_at = idx + 1
    new_lines = list(lines)
    if insert_at > 0 and insert_at <= len(new_lines) and new_lines[insert_at - 1].strip():
        pass
    new_lines.insert(insert_at, f"    {key}: {value}\n")
    return new_lines


def sync_ros2_params_to_config(config_path: str,
                               dx: float, dy: float,
                               boresight_path: str,
                               bx: float, by: float, z_ref: float) -> bool:
    if not config_path or not os.path.exists(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as exc:
        print(f"[同步] 读 ROS2 参数失败: {config_path}: {exc}")
        return False

    updates = {
        "boresight_file": f"\"{_path_for_yaml(boresight_path)}\"",
        "pixel_offset_x": f"{dx:.2f}",
        "pixel_offset_y": f"{dy:.2f}",
    }
    if abs(bx) > 1e-9 or abs(by) > 1e-9:
        updates["baseline_bx"] = f"{bx:.6g}"
        updates["baseline_by"] = f"{by:.6g}"
    if z_ref > 0.0:
        updates["baseline_z_ref"] = f"{z_ref:.6g}"

    new_lines = list(lines)
    for key, value in updates.items():
        new_lines, replaced = _replace_scalar_yaml_key(new_lines, key, value)
        if not replaced:
            new_lines = _insert_ros2_param(new_lines, key, value)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return True
    except OSError as exc:
        print(f"[同步] 写 ROS2 参数失败: {config_path}: {exc}")
        return False


def save_and_sync_all(settings: CalibratorSettings,
                      u_L: float,
                      v_L: float,
                      frame_w: int,
                      frame_h: int,
                      bx: float,
                      by: float,
                      z_ref: float,
                      reason: str) -> bool:
    ok = save_boresight(
        settings.output_path,
        u_L, v_L, frame_w, frame_h,
        bx, by, z_ref,
    )
    synced = False
    ros2_synced: List[Tuple[str, bool]] = []
    dx = u_L - frame_w * 0.5
    dy = v_L - frame_h * 0.5
    if ok and settings.sync_config:
        synced = sync_pixel_offset_to_config(settings.config_path, dx, dy)
    if ok and settings.sync_ros2_config:
        for cfg_path in settings.ros2_config_paths or []:
            ros2_synced.append((
                cfg_path,
                sync_ros2_params_to_config(
                    cfg_path,
                    dx,
                    dy,
                    settings.output_path,
                    bx,
                    by,
                    z_ref,
                ),
            ))
    if ok:
        msg = f"[保存:{reason}] OK → {settings.output_path}"
        if settings.sync_config:
            msg += f"  | pixel_offset {'sync OK' if synced else 'sync FAIL'}"
        if settings.sync_ros2_config and ros2_synced:
            ok_count = sum(1 for _, is_ok in ros2_synced if is_ok)
            msg += f"  | ros2_params {ok_count}/{len(ros2_synced)}"
        print(msg)
        for cfg_path, is_ok in ros2_synced:
            print(f"[同步] ROS2参数 {'OK' if is_ok else 'FAIL'}: {cfg_path}")
        return True
    print(f"[保存:{reason}] 失败，检查路径/权限")
    return False


def default_ros2_config_paths() -> List[str]:
    candidates = [
        os.path.join(PROJECT_ROOT, "radar_gimbal_gazebo", "config", "lasertracking_gimbal_detect_params.yaml"),
        os.path.join(PROJECT_ROOT, "radar_gimbal_gazebo", "config", "lasertracking_hw_params.yaml"),
    ]
    return [p for p in candidates if os.path.exists(p)]


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(val, hi))


def main(argv: Optional[List[str]] = None) -> int:
    global _EXTRA_MVIMPORT_ROOTS

    parser = argparse.ArgumentParser(description="Boresight calibrator")
    parser.add_argument(
        "--config",
        default=os.path.join(LASERTRACKING_DIR, "tracking_system_old", "config_tracking.yaml"),
        help="tracking 主配置路径",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(CURRENT_DIR, "boresight.yaml"),
        help="boresight 保存路径",
    )
    parser.add_argument("--no-detector", action="store_true", help="不加载 YOLO 检测器")
    parser.add_argument("--no-sync-config", action="store_true",
                        help="关闭自动同步到 config_tracking.yaml")
    parser.add_argument("--no-sync-ros2-config", action="store_true",
                        help="关闭自动同步到 ROS2 参数 yaml")
    parser.add_argument("--no-auto-save-on-exit", action="store_true",
                        help="退出程序时不自动保存")
    parser.add_argument("--ros2-config", action="append", default=None,
                        help="额外指定要同步的 ROS2 参数 yaml，可重复传入")
    parser.add_argument("--check-config", action="store_true",
                        help="只检查配置/路径，不打开相机")
    parser.add_argument("--step", type=int, default=2, help="初始步长(px)")
    parser.add_argument("--window-size", default="1280x720", help="显示窗口尺寸 WxH")
    parser.add_argument(
        "--mvimport-path",
        default="",
        help="海康 MvImport 的父目录（其下应有子目录 MvImport/），也可用环境变量 MVIMPORT_PATH",
    )
    parser.add_argument(
        "--ros-image-topic",
        default="",
        help=(
            "ROS2 sensor_msgs/Image 话题取图（需 source ROS 且驱动在发布），与 "
            "radar_gimbal_gazebo/config/lasertracking_hw_params.yaml 里 "
            "tele_image_topic 一致即可，实机长焦常为 /gimbal_camera/image_raw"
        ),
    )
    args = parser.parse_args(argv)

    _EXTRA_MVIMPORT_ROOTS.clear()
    mp = str(args.mvimport_path or "").strip()
    if mp:
        _EXTRA_MVIMPORT_ROOTS.append(os.path.abspath(mp))

    try:
        w_str, h_str = args.window_size.lower().split("x")
        win_w = max(320, int(w_str))
        win_h = max(240, int(h_str))
    except Exception:
        win_w, win_h = 1280, 720

    ros2_config_paths = default_ros2_config_paths()
    if args.ros2_config:
        ros2_config_paths.extend(os.path.abspath(p) for p in args.ros2_config)
    ros2_config_paths = list(dict.fromkeys(ros2_config_paths))

    settings = CalibratorSettings(
        config_path=os.path.abspath(args.config),
        output_path=os.path.abspath(args.output),
        detector_enabled=not args.no_detector,
        sync_config=not args.no_sync_config,
        sync_ros2_config=not args.no_sync_ros2_config,
        auto_save_on_exit=not args.no_auto_save_on_exit,
        ros2_config_paths=ros2_config_paths,
        step_px=max(1, min(50, int(args.step))),
        window_size=(win_w, win_h),
    )

    ros_topic = str(args.ros_image_topic or "").strip()

    print(f"config: {settings.config_path}")
    print(f"output: {settings.output_path}")
    if ros_topic:
        print(f"相机: ROS 话题 {ros_topic}")
    elif mp:
        print(f"MvImport 搜索根目录（额外）: {os.path.abspath(mp)}")
    print(f"detector: {'启用' if settings.detector_enabled else '禁用'}   "
          f"sync_config: {'启用' if settings.sync_config else '禁用'}   "
          f"sync_ros2_config: {'启用' if settings.sync_ros2_config else '禁用'}")

    try:
        config = load_config(settings.config_path)
    except OSError as exc:
        print(f"读 config 失败: {exc}")
        return 1

    if args.check_config:
        return print_config_check(config, settings)

    bx, by, z_ref = read_baseline(config)
    fx, fy = read_intrinsics(config)
    has_parallax_model = (fx > 0.0 and fy > 0.0 and (abs(bx) > 1e-9 or abs(by) > 1e-9))

    detector = build_detector(config) if settings.detector_enabled else None
    if detector is None:
        print("提示: 无检测器，请手动用鼠标对齐激光点（c 键不可用）")

    try:
        camera = build_camera(config, ros_image_topic=ros_topic or None)
    except RuntimeError as exc:
        print(f"相机打开失败: {exc}")
        return 2

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
    cv2.moveWindow(WINDOW_NAME, 80, 80)
    mouse_state = MouseState()
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, mouse_state)

    u_L: Optional[float] = None
    v_L: Optional[float] = None
    init_u: Optional[float] = None
    init_v: Optional[float] = None
    last_action = "none"
    last_key_raw = -1
    frame_w = 0
    frame_h = 0
    show_parallax_overlay = False
    parallax_distances = (5.0, 15.0, 30.0, 60.0)

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_h, frame_w = frame.shape[:2]
            if u_L is None:
                u_L = frame_w * 0.5
                v_L = frame_h * 0.5
            if init_u is None:
                init_u, init_v = u_L, v_L

            target_center = None
            target_bbox = None
            target_conf = 0.0
            if detector is not None:
                best = detect_best_target(detector, frame)
                if best is not None:
                    target_bbox, target_center, target_conf = best

            view = frame.copy()
            if target_bbox is not None:
                x1, y1, x2, y2 = target_bbox
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                draw_crosshair(view,
                               (int(round(target_center[0])),
                                int(round(target_center[1]))),
                               (0, 255, 0))
            draw_crosshair(view, (int(round(u_L)), int(round(v_L))), (0, 0, 255))

            if show_parallax_overlay and has_parallax_model:
                for z in parallax_distances:
                    du, dv = parallax_offset(fx, fy, bx, by, z,
                                             z_ref=z_ref if z_ref > 0 else None)
                    pred_x = int(round(u_L + du))
                    pred_y = int(round(v_L + dv))
                    if 0 <= pred_x < frame_w and 0 <= pred_y < frame_h:
                        cv2.circle(view, (pred_x, pred_y), 14, (0, 165, 255), 2, cv2.LINE_AA)
                        cv2.putText(view, f"z={z:g}m",
                                    (pred_x + 16, pred_y + 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 165, 255), 2, cv2.LINE_AA)

            du = u_L - init_u
            dv = v_L - init_v
            if detector is None:
                status_str = "NO DETECTOR (manual mode)"
            elif target_center is not None:
                status_str = "TARGET LOCKED"
            else:
                status_str = "NO TARGET (put UAV / laser-module in view, or use --no-detector)"
            lines = [
                f"STATUS: {status_str}",
                f"Boresight u_L={u_L:.2f}  v_L={v_L:.2f}",
                f"Offset du={du:+.2f}  dv={dv:+.2f}  step={settings.step_px}",
                f"Frame={frame_w}x{frame_h}"
                f"{'  parallax_overlay=on' if show_parallax_overlay else ''}",
            ]
            if target_center is not None:
                lines.append(
                    f"Target center=({target_center[0]:.1f},{target_center[1]:.1f}) "
                    f"conf={target_conf:.2f}"
                )
            if has_parallax_model:
                lines.append(f"Baseline bx={bx:.3f} by={by:.3f} z_ref={z_ref:g} (fx/fy from config)")
            lines.append(f"Last key={last_key_raw} action={last_action}")
            lines.append("Keys: arrows/WASD move  [/-/]/+step  c=center i=init r=reset s=save p=parallax q=quit")
            lines.append("Mouse: L=move  R=set_init  Wheel=step")
            draw_text_lines(view, lines)

            if view.shape[1] != win_w or view.shape[0] != win_h:
                display = cv2.resize(view, (win_w, win_h))
            else:
                display = view
            cv2.imshow(WINDOW_NAME, display)
            with mouse_state.lock:
                mouse_state.view_w = frame_w
                mouse_state.view_h = frame_h

            key_raw = cv2.waitKeyEx(1)
            if key_raw >= 0:
                last_key_raw = key_raw
            key_low = key_raw & 0xFF if key_raw >= 0 else -1

            arrow_hit = None
            if key_raw in ARROW_LEFT:
                arrow_hit = "left"
            elif key_raw in ARROW_RIGHT:
                arrow_hit = "right"
            elif key_raw in ARROW_UP:
                arrow_hit = "up"
            elif key_raw in ARROW_DOWN:
                arrow_hit = "down"

            if arrow_hit == "left":
                u_L -= settings.step_px
                last_action = "move_left"
            elif arrow_hit == "right":
                u_L += settings.step_px
                last_action = "move_right"
            elif arrow_hit == "up":
                v_L -= settings.step_px
                last_action = "move_up"
            elif arrow_hit == "down":
                v_L += settings.step_px
                last_action = "move_down"
            elif key_low in (27, ord("q"), ord("Q")):
                last_action = "quit"
                break
            elif key_low in (ord("a"), ord("A")):
                u_L -= settings.step_px
                last_action = "move_left(wasd)"
            elif key_low in (ord("d"), ord("D")):
                u_L += settings.step_px
                last_action = "move_right(wasd)"
            elif key_low in (ord("w"), ord("W")):
                v_L -= settings.step_px
                last_action = "move_up(wasd)"
            elif key_low in (ord("x"), ord("X")):
                v_L += settings.step_px
                last_action = "move_down(wasd)"
            elif key_low in (ord("["), ord("-")):
                settings.step_px = max(1, settings.step_px - 1)
                last_action = f"step_down({settings.step_px})"
            elif key_low in (ord("]"), ord("+"), ord("=")):
                settings.step_px = min(200, settings.step_px + 1)
                last_action = f"step_up({settings.step_px})"
            elif key_low in (ord("r"), ord("R")):
                u_L, v_L = init_u, init_v
                last_action = "reset"
            elif key_low in (ord("c"), ord("C")) and target_center is not None:
                u_L, v_L = float(target_center[0]), float(target_center[1])
                init_u, init_v = u_L, v_L
                last_action = "center_on_target"
            elif key_low in (ord("i"), ord("I")):
                init_u, init_v = u_L, v_L
                last_action = "set_init"
            elif key_low in (ord("p"), ord("P")):
                if has_parallax_model:
                    show_parallax_overlay = not show_parallax_overlay
                    last_action = f"parallax_overlay={'on' if show_parallax_overlay else 'off'}"
                else:
                    last_action = "parallax_overlay_NA(no baseline)"
            elif key_low in (ord("s"), ord("S")):
                ok = save_and_sync_all(
                    settings=settings,
                    u_L=u_L,
                    v_L=v_L,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    bx=bx,
                    by=by,
                    z_ref=z_ref,
                    reason="manual",
                )
                last_action = "save_ok" if ok else "save_fail"

            u_L = clamp(u_L, 0.0, float(frame_w - 1))
            v_L = clamp(v_L, 0.0, float(frame_h - 1))

            with mouse_state.lock:
                if mouse_state.pending:
                    ev = mouse_state.event
                    mx = mouse_state.x
                    my = mouse_state.y
                    mflags = mouse_state.flags
                    vw = mouse_state.view_w
                    vh = mouse_state.view_h
                    mouse_state.pending = False
                else:
                    ev = -1
                    mx = my = 0
                    mflags = 0
                    vw = vh = 0

            if ev >= 0 and vw > 0 and vh > 0:
                sx = vw / float(win_w)
                sy = vh / float(win_h)
                img_x = mx * sx
                img_y = my * sy
                if ev == cv2.EVENT_LBUTTONDOWN:
                    u_L = clamp(img_x, 0.0, float(vw - 1))
                    v_L = clamp(img_y, 0.0, float(vh - 1))
                    last_action = "mouse_move"
                elif ev == cv2.EVENT_RBUTTONDOWN:
                    init_u, init_v = u_L, v_L
                    last_action = "mouse_set_init"
                elif ev == cv2.EVENT_MOUSEWHEEL:
                    try:
                        delta = cv2.getMouseWheelDelta(mflags)
                    except Exception:
                        delta = 1 if mflags > 0 else -1
                    if delta > 0:
                        settings.step_px = min(200, settings.step_px + 1)
                        last_action = f"mouse_step_up({settings.step_px})"
                    elif delta < 0:
                        settings.step_px = max(1, settings.step_px - 1)
                        last_action = f"mouse_step_down({settings.step_px})"

    except KeyboardInterrupt:
        print("\n用户中断 (Ctrl+C)")
    finally:
        if (settings.auto_save_on_exit and u_L is not None and v_L is not None
                and frame_w > 0 and frame_h > 0):
            save_and_sync_all(
                settings=settings,
                u_L=u_L,
                v_L=v_L,
                frame_w=frame_w,
                frame_h=frame_h,
                bx=bx,
                by=by,
                z_ref=z_ref,
                reason="exit",
            )
        try:
            camera.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("已退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
