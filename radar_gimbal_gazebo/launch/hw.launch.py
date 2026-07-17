from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import os

# conda 下需让 Python 先搜 Humble 的 site-packages（Ubuntu 22.04 + Humble 为 3.10）
_HUMBLE_SITE = "/opt/ros/humble/lib/python3.10/site-packages"


def _ros_python_nodes_env():
    """确保 ROS Python 包优先于 conda。"""
    path = "/usr/bin:/bin:/opt/ros/humble/bin:/usr/local/bin"
    py_path = os.environ.get("PYTHONPATH", "").strip()
    if py_path:
        merged = _HUMBLE_SITE + os.pathsep + py_path
    else:
        merged = _HUMBLE_SITE
    return {"PATH": path, "PYTHONPATH": merged}


def generate_launch_description():
    pkg_this = get_package_share_directory("radar_gimbal_gazebo")
    pkg_this_path = Path(pkg_this)

    tracker_sim_params = (pkg_this_path / "config" / "lasertracking_sim_params.yaml").resolve()
    tracker_detect_params = (
        pkg_this_path / "config" / "lasertracking_gimbal_detect_params_common.yaml"
    ).resolve()
    tracker_hw_params = (pkg_this_path / "config" / "lasertracking_hw_params.yaml").resolve()

    for cfg in (tracker_sim_params, tracker_detect_params, tracker_hw_params):
        if not cfg.is_file():
            raise RuntimeError(f"未找到配置文件: {cfg}")

    def _hw_group(context, *args, **kwargs):
        return [
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
            ),
        ]

    return LaunchDescription(
        [
            OpaqueFunction(function=_hw_group),
        ]
    )
