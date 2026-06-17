1. 本目录为 SolidWorks 导出参考，勿用 colcon 编译（已含 COLCON_IGNORE）。
2. urdf/雷达.SLDASM.urdf 中网格为相对路径 ../meshes/*.STL，可在本目录下用 RViz/check_urdf 粗查。
3. Gazebo 仿真请使用 ROS2 包 radar_gimbal_gazebo：
     ros2 launch radar_gimbal_gazebo sim.launch.py
   launch 会把网格改成绝对 file://，避免 Gazebo 无法解析 package:// 而不显示模型。
4. 从 SW 重新导出后：同步替换 radar_gimbal_gazebo/urdf/radar_gimbal.urdf.xacro 内 macro，
   并把 meshes/*.STL 复制到 radar_gimbal_gazebo/meshes/。
