from pathlib import Path
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# conda 下需让 Python 先搜 Humble 的 site-packages（Ubuntu 22.04 + Humble 为 3.10）
_HUMBLE_SITE = "/opt/ros/humble/lib/python3.10/site-packages"


def _ros_python_nodes_env():
    path = "/usr/bin:/bin:/opt/ros/humble/bin:/usr/local/bin"
    py_path = os.environ.get("PYTHONPATH", "").strip()
    merged = _HUMBLE_SITE + os.pathsep + py_path if py_path else _HUMBLE_SITE
    return {"PATH": path, "PYTHONPATH": merged}


def _ld_library_path_for_ros_cpp() -> str:
    """避免 Conda 的 libstdc++ 抢先加载，导致 librclcpp 需要 GLIBCXX_3.4.30 却找不到。"""
    system_lib = "/usr/lib/x86_64-linux-gnu"
    parts = []
    if os.path.isdir(system_lib):
        parts.append(system_lib)
    for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not p:
            continue
        lp = p.lower()
        if any(x in lp for x in ("anaconda", "miniconda", "mambaforge", "micromamba")):
            continue
        if p not in parts:
            parts.append(p)
    return os.pathsep.join(parts)


def _hik_cpp_driver_env():
    """海康驱动为 C++ 节点：使用系统 libstdc++，并保留非 Conda 的 LD 路径。"""
    env = os.environ.copy()
    ld = _ld_library_path_for_ros_cpp()
    if ld:
        env["LD_LIBRARY_PATH"] = ld
    return env


def _load_hik_exposure_overrides_from_hw(hw_yaml: Path) -> tuple[dict, dict]:
    """从 lasertracking_hw_params.yaml 读取 hw_hik_*，作为海康节点参数覆盖（只改 hw 即可同步两路相机）。"""
    with open(hw_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    p = (data.get("lasertracking_tracker") or {}).get("ros__parameters") or {}

    def _num(key: str, default: float) -> float:
        v = p.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    wide_exp = int(round(_num("hw_hik_wide_exposure_time", 39000.0)))
    wide_gain = float(_num("hw_hik_wide_gain", 16.0))
    tele_exp = int(round(_num("hw_hik_tele_exposure_time", 35000.0)))
    tele_gain = float(_num("hw_hik_tele_gain", 20.0))

    wide_ov = {"exposure_time": wide_exp, "gain": wide_gain}
    tele_ov = {"exposure_time": tele_exp, "gain": tele_gain}
    return wide_ov, tele_ov


def generate_launch_description():
    pkg_this = get_package_share_directory("radar_gimbal_gazebo")
    pkg_this_path = Path(pkg_this)

    tracker_sim_params = (pkg_this_path / "config" / "lasertracking_sim_params.yaml").resolve()
    tracker_detect_params = (
        pkg_this_path / "config" / "lasertracking_gimbal_detect_params.yaml"
    ).resolve()
    tracker_hw_params = (pkg_this_path / "config" / "lasertracking_hw_params.yaml").resolve()
    default_wide_params = (pkg_this_path / "config" / "hik_wide.yaml").resolve()
    default_tele_params = (pkg_this_path / "config" / "hik_tele.yaml").resolve()

    for cfg in (tracker_sim_params, tracker_detect_params, tracker_hw_params):
        if not cfg.is_file():
            raise RuntimeError(f"未找到配置文件: {cfg}")

    def _truthy(lc: LaunchConfiguration, context) -> bool:
        return lc.perform(context).strip().lower() in ("1", "true", "yes", "on")

    def _group(context, *args, **kwargs):
        wide_params_file = LaunchConfiguration("wide_params_file").perform(context).strip()
        tele_params_file = LaunchConfiguration("tele_params_file").perform(context).strip()
        wide_hik_ov, tele_hik_ov = _load_hik_exposure_overrides_from_hw(tracker_hw_params)

        wide_param_list = []
        if wide_params_file:
            wide_param_list.append(wide_params_file)
        wide_param_list.append(wide_hik_ov)

        tele_param_list = []
        if tele_params_file:
            tele_param_list.append(tele_params_file)
        tele_param_list.append(tele_hik_ov)
        tele_serial = LaunchConfiguration("tele_device_serial").perform(context).strip()
        if tele_serial:
            tele_param_list.append({"device_serial": tele_serial})
        tele_di = LaunchConfiguration("tele_device_index").perform(context).strip()
        if tele_di != "":
            tele_param_list.append({"device_index": int(tele_di)})
        wide_serial = LaunchConfiguration("wide_device_serial").perform(context).strip()
        if wide_serial:
            wide_param_list.append({"device_serial": wide_serial})
        wide_di = LaunchConfiguration("wide_device_index").perform(context).strip()
        if wide_di != "":
            wide_param_list.append({"device_index": int(wide_di)})

        start_cameras = _truthy(LaunchConfiguration("start_cameras"), context)
        start_wide = _truthy(LaunchConfiguration("start_wide_camera"), context)
        start_tele = _truthy(LaunchConfiguration("start_tele_camera"), context)
        start_tracker = _truthy(LaunchConfiguration("start_tracker"), context)

        nodes = []
        if start_cameras and start_wide:
            nodes.append(
                Node(
                    package=LaunchConfiguration("sdk_package"),
                    executable=LaunchConfiguration("sdk_executable"),
                    namespace="fixed_camera",
                    name=LaunchConfiguration("wide_node_name"),
                    output="screen",
                    parameters=wide_param_list,
                    additional_env=_hik_cpp_driver_env(),
                )
            )
        if start_cameras and start_tele:
            nodes.append(
                Node(
                    package=LaunchConfiguration("sdk_package"),
                    executable=LaunchConfiguration("sdk_executable"),
                    namespace="gimbal_camera",
                    name=LaunchConfiguration("tele_node_name"),
                    output="screen",
                    parameters=tele_param_list,
                    additional_env=_hik_cpp_driver_env(),
                )
            )
        if start_tracker:
            nodes.append(
                Node(
                    package="radar_gimbal_gazebo",
                    executable="lasertracking_tracker",
                    output="screen",
                    parameters=[
                        str(tracker_sim_params),
                        str(tracker_detect_params),
                        str(tracker_hw_params),
                    ],
                    additional_env=_ros_python_nodes_env(),
                )
            )
        return nodes

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_cameras", default_value="true"),
            DeclareLaunchArgument(
                "start_tracker",
                default_value="true",
                description="是否启动 lasertracking_tracker；标定 boresight 时可设 false，仅起海康节点",
            ),
            DeclareLaunchArgument(
                "start_wide_camera",
                default_value="false",
                description="start_cameras 为 true 时是否启动短焦 fixed_camera；默认 false 仅开长焦 gimbal_camera",
            ),
            DeclareLaunchArgument(
                "start_tele_camera",
                default_value="true",
                description="start_cameras 为 true 时是否启动长焦 gimbal_camera",
            ),
            DeclareLaunchArgument("sdk_package", default_value="hik_camera_ros2_driver"),
            DeclareLaunchArgument("sdk_executable", default_value="hik_camera_ros2_driver_node"),
            DeclareLaunchArgument("wide_node_name", default_value="wide_camera"),
            DeclareLaunchArgument("tele_node_name", default_value="tele_camera"),
            DeclareLaunchArgument("wide_params_file", default_value=str(default_wide_params)),
            DeclareLaunchArgument("tele_params_file", default_value=str(default_tele_params)),
            DeclareLaunchArgument(
                "tele_device_serial",
                default_value="",
                description="非空则覆盖 hik_tele.yaml：按序列号子串选长焦（双机推荐，避免 device_index 漂移）",
            ),
            DeclareLaunchArgument(
                "tele_device_index",
                default_value="",
                description="非空则覆盖 hik_tele.yaml 的 device_index（整数）；与 tele_device_serial 二选一或 serial 优先",
            ),
            DeclareLaunchArgument(
                "wide_device_serial",
                default_value="",
                description="start_wide_camera:=true 时可选：短焦序列号子串",
            ),
            DeclareLaunchArgument(
                "wide_device_index",
                default_value="",
                description="start_wide_camera:=true 时可选：覆盖 hik_wide.yaml 的 device_index",
            ),
            OpaqueFunction(function=_group),
        ]
    )
