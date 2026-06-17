from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
import xacro

# Gazebo Classic 将 URDF 转 SDF 时常无法解析 package://，导致网格不显示；改为绝对 file://
_MESH_STL = [
    "base_link.STL",
    "fix_camera_one_Link.STL",
    "camera_one_optical_Link.STL",
    "yaw_joint_Link.STL",
    "pich_joint_Link.STL",
    "movable_camera_tow_Link.STL",
    "camera_tow_optical_Link.STL",
]


def _meshes_package_to_file_uri(urdf_xml: str, pkg_share: str) -> str:
    meshes = Path(pkg_share).resolve() / "meshes"
    for stl in _MESH_STL:
        src = f"package://radar_gimbal_gazebo/meshes/{stl}"
        dst_path = meshes / stl
        if not dst_path.is_file():
            raise FileNotFoundError(
                f"Gazebo 需要网格文件: {dst_path}"
            )
        urdf_xml = urdf_xml.replace(src, "file://" + str(dst_path))
    return urdf_xml


_GZ_SCRIPT_URI = "file://media/materials/scripts/gazebo.material"


def _drone_urdf_path_for_gazebo(repo_root: Path, trim_stack_base_z: str) -> str:
    """xacro 展开 + package://drone/meshes → file://…/aerial.SLDASM/meshes，供 gz sdf / Gazebo 解析。"""
    xacro_path = repo_root / "aerial.SLDASM" / "urdf" / "drone.urdf.xacro"
    text = xacro.process_file(
        str(xacro_path),
        mappings={"trim_stack_base_z": trim_stack_base_z.strip()},
    ).toxml()
    text = re.sub(r'^\s*<\?xml[^>]*\?>\s*', "", text, count=1)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    meshes = (repo_root / "aerial.SLDASM" / "meshes").resolve()
    text = text.replace("package://drone/meshes/", "file://" + str(meshes) + "/")
    fd, tmp = tempfile.mkstemp(prefix="drone_gazebo_", suffix=".urdf")
    os.close(fd)
    Path(tmp).write_text(text, encoding="utf-8")
    return tmp


def _inject_drone_visual_materials_sdf(sdf_root: ET.Element) -> None:
    """Gazebo URDF→SDF 合并多 visual 后，仅第一个能带上 URDF 里写的 gazebo 材质；在 SDF 上逐 visual 补全。"""
    for link in sdf_root.iter("link"):
        if link.get("name") != "base_link":
            continue
        for visual in link.findall("visual"):
            vname = visual.get("name") or ""
            if "trim_mid" in vname:
                gz_mat = "Gazebo/Red"
            else:
                gz_mat = "Gazebo/Black"
            old = visual.find("material")
            if old is not None:
                visual.remove(old)
            mat = ET.SubElement(visual, "material")
            script = ET.SubElement(mat, "script")
            sn = ET.SubElement(script, "name")
            sn.text = gz_mat
            su = ET.SubElement(script, "uri")
            su.text = _GZ_SCRIPT_URI


def _drone_sdf_path_for_spawn(repo_root: Path, trim_stack_base_z: str) -> str:
    """URDF → gz sdf -p → 注入材质 → 无 XML 头的 SDF 文件（避免 spawn_entity lxml 报错）。"""
    urdf_path = _drone_urdf_path_for_gazebo(repo_root, trim_stack_base_z)
    proc = subprocess.run(
        ["/usr/bin/gz", "sdf", "-p", urdf_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "gz sdf -p 失败（无法生成无人机 SDF）: "
            + (proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}")
        )
    root = ET.fromstring(proc.stdout)
    _inject_drone_visual_materials_sdf(root)
    body = ET.tostring(root, encoding="unicode")
    fd, sdf_path = tempfile.mkstemp(prefix="drone_spawn_", suffix=".sdf")
    os.close(fd)
    Path(sdf_path).write_text(body, encoding="utf-8")
    return sdf_path


# 不自定义 GAZEBO_MASTER_URI，与 gzserver/gzclient 共用默认 11345，避免 spawn 进错 master。

# conda 下需让 Python 先搜 Humble 的 site-packages（Ubuntu 22.04 + Humble 为 3.10）
_HUMBLE_SITE = "/opt/ros/humble/lib/python3.10/site-packages"


def _ros_python_nodes_env():
    """spawn_entity / spawner：系统 python3 + ROS 包优先于 conda。"""
    path = "/usr/bin:/bin:/opt/ros/humble/bin:/usr/local/bin"
    py_path = os.environ.get("PYTHONPATH", "").strip()
    if py_path:
        merged = _HUMBLE_SITE + os.pathsep + py_path
    else:
        merged = _HUMBLE_SITE
    return {"PATH": path, "PYTHONPATH": merged}


def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    pkg_this = get_package_share_directory("radar_gimbal_gazebo")
    repo_root = Path(__file__).resolve().parents[2]
    pkg_this_path = Path(pkg_this)

    sim_launch_params_file = pkg_this_path / "config" / "sim_launch_params.yaml"
    if not sim_launch_params_file.is_file():
        raise RuntimeError(f"未找到仿真配置文件: {sim_launch_params_file}")
    with sim_launch_params_file.open("r", encoding="utf-8") as f:
        sim_cfg_root = yaml.safe_load(f) or {}
    sim_cfg = sim_cfg_root.get("sim_launch", {})
    if not isinstance(sim_cfg, dict):
        raise RuntimeError("sim_launch_params.yaml 格式错误: 缺少 sim_launch 字典")

    trim_stack_base_z = str(sim_cfg.get("trim_stack_base_z", 0.610))
    enable_lasertracking = bool(sim_cfg.get("enable_lasertracking", False))
    motion_cfg = sim_cfg.get("drone_random_motion", {})
    if not isinstance(motion_cfg, dict):
        raise RuntimeError("sim_launch_params.yaml 格式错误: drone_random_motion 必须是字典")

    motion_enabled = bool(motion_cfg.get("enabled", True))
    motion_center_x = float(motion_cfg.get("center_x", 20.0))
    motion_center_y = float(motion_cfg.get("center_y", 0.0))
    motion_center_z = float(motion_cfg.get("center_z", 1.0))
    motion_axis = str(motion_cfg.get("axis", "x")).strip()
    motion_half_range = float(motion_cfg.get("half_range_m", 1.5))
    motion_v_min = float(motion_cfg.get("speed_min", 0.2))
    motion_v_max = float(motion_cfg.get("speed_max", 0.8))
    motion_speed_jitter = float(motion_cfg.get("speed_jitter_mps", 0.12))
    motion_hh = float(motion_cfg.get("height_half_range_m", 0.5))
    motion_hv_min = float(motion_cfg.get("height_speed_min", -1.0))
    motion_hv_max = float(motion_cfg.get("height_speed_max", -1.0))
    motion_hj = float(motion_cfg.get("height_speed_jitter_mps", -1.0))
    motion_h_stop = float(motion_cfg.get("height_stop_prob", 0.03))
    motion_h_rev = float(motion_cfg.get("height_reverse_prob", 0.02))
    motion_h_pause_min = float(motion_cfg.get("height_pause_min_s", 0.15))
    motion_h_pause_max = float(motion_cfg.get("height_pause_max_s", 0.6))

    lasertracking_sim_params_file = (pkg_this_path / "config" / "lasertracking_sim_params.yaml").resolve()
    lasertracking_gimbal_detect_params_file = (
        pkg_this_path / "config" / "lasertracking_gimbal_detect_params.yaml"
    ).resolve()

    world = os.path.join(pkg_this, "worlds", "empty.world")
    xacro_file = os.path.join(pkg_this, "urdf", "radar_gimbal.urdf.xacro")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gazebo.launch.py")
        ),
        launch_arguments={"world": world}.items(),
    )

    robot_description_xml = xacro.process_file(xacro_file).toxml()
    robot_description_xml = re.sub(r"<!--.*?-->", "", robot_description_xml, flags=re.DOTALL)
    robot_description_xml = _meshes_package_to_file_uri(robot_description_xml, pkg_this)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": ParameterValue(robot_description_xml, value_type=str)}],
    )

    spawn_entity = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                arguments=[
                    "-topic",
                    "robot_description",
                    "-entity",
                    "radar_gimbal",
                    "-z",
                    "0.5",
                    "-timeout",
                    "120.0",
                ],
                output="screen",
                additional_env=_ros_python_nodes_env(),
            ),
        ],
    )

    def _spawn_drone(context, *args, **kwargs):
        sdf_path = _drone_sdf_path_for_spawn(repo_root, trim_stack_base_z)
        return [
            TimerAction(
                period=7.0,
                actions=[
                    Node(
                        package="gazebo_ros",
                        executable="spawn_entity.py",
                        arguments=[
                            "-file",
                            sdf_path,
                            "-entity",
                            "drone",
                            "-x",
                            "20.0",
                            "-y",
                            "0.0",
                            "-z",
                            "1.0",
                            "-timeout",
                            "120.0",
                        ],
                        output="screen",
                        additional_env=_ros_python_nodes_env(),
                    ),
                ],
            )
        ]

    spawn_aerial = OpaqueFunction(function=_spawn_drone)

    def _drone_motion_group(context, *args, **kwargs):
        if not motion_enabled:
            return []
        return [
            TimerAction(
                period=10.0,
                actions=[
                    Node(
                        package="radar_gimbal_gazebo",
                        executable="drone_random_motion",
                        parameters=[
                            {
                                "model_name": "drone",
                                "center_x": motion_center_x,
                                "center_y": motion_center_y,
                                "center_z": motion_center_z,
                                "axis": motion_axis,
                                "half_range_m": motion_half_range,
                                "speed_min": motion_v_min,
                                "speed_max": motion_v_max,
                                "speed_jitter_mps": motion_speed_jitter,
                                "height_half_range_m": motion_hh,
                                "height_speed_min": motion_hv_min,
                                "height_speed_max": motion_hv_max,
                                "height_speed_jitter_mps": motion_hj,
                                "height_stop_prob": motion_h_stop,
                                "height_reverse_prob": motion_h_rev,
                                "height_pause_min_s": motion_h_pause_min,
                                "height_pause_max_s": motion_h_pause_max,
                                "publish_rate": 30.0,
                            }
                        ],
                        output="screen",
                        additional_env=_ros_python_nodes_env(),
                    ),
                ],
            )
        ]

    drone_motion = OpaqueFunction(function=_drone_motion_group)

    spawn_controllers = TimerAction(
        period=12.0,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "joint_state_broadcaster",
                    "gimbal_position_controller_hw",
                    "-c",
                    # URDF 里 gazebo_ros2_control 未写 <ros><namespace> 时，gazebo_ros 默认 ns 为 "/"，
                    # controller_manager 节点名为 /controller_manager（与 spawn -entity 名无关）。
                    "/controller_manager",
                    "--controller-manager-timeout",
                    "120",
                ],
                output="screen",
                additional_env=_ros_python_nodes_env(),
            ),
        ],
    )

    # 上层依旧发布到 /gimbal_position_controller/commands，限坡后再送到 *_hw 控制器
    command_slew_limiter = TimerAction(
        period=12.5,
        actions=[
            Node(
                package="radar_gimbal_gazebo",
                executable="gimbal_command_slew_limiter",
                parameters=[
                    {
                        "input_topic": "/gimbal_position_controller/commands",
                        "output_topic": "/gimbal_position_controller_hw/commands",
                        "update_rate_hz": 500.0,
                        # 快速响应版限坡：保持平滑，同时避免“慢吞吞”
                        "yaw_acc": 35.0,
                        "yaw_dec": 45.0,
                        "yaw_vmax": 30.0,
                        "pitch_acc": 60.0,
                        "pitch_dec": 80.0,
                        "pitch_vmax": 30.0,
                        "tracking_kp": 20.0,
                    }
                ],
                output="screen",
                additional_env=_ros_python_nodes_env(),
            ),
        ],
    )

    def _lasertracking_group(context, *args, **kwargs):
        if not enable_lasertracking:
            return []
        sim_params_file = Path(lasertracking_sim_params_file).expanduser()
        gimbal_detect_params_file = Path(lasertracking_gimbal_detect_params_file).expanduser()
        if not sim_params_file.is_file():
            raise RuntimeError(f"未找到 lasertracking_sim_params_file: {sim_params_file}")
        if not gimbal_detect_params_file.is_file():
            raise RuntimeError(
                f"未找到 lasertracking_gimbal_detect_params_file: {gimbal_detect_params_file}"
            )
        return [
            TimerAction(
                period=13.0,
                actions=[
                    Node(
                        package="radar_gimbal_gazebo",
                        executable="gimbal_comm_bridge",
                        output="screen",
                        additional_env=_ros_python_nodes_env(),
                    ),
                    Node(
                        package="radar_gimbal_gazebo",
                        executable="lasertracking_tracker",
                        output="screen",
                        parameters=[
                            str(sim_params_file),
                            str(gimbal_detect_params_file),
                        ],
                        additional_env=_ros_python_nodes_env(),
                    ),
                ],
            )
        ]

    lasertracking = OpaqueFunction(function=_lasertracking_group)

    return LaunchDescription(
        [
            gazebo,
            robot_state_publisher,
            spawn_entity,
            spawn_aerial,
            drone_motion,
            spawn_controllers,
            command_slew_limiter,
            lasertracking,
        ]
    )
