#!/usr/bin/env bash
# WSL 仿真一键启动脚本
# 用法：在 WSL Ubuntu-22.04 中执行 bash /mnt/e/1/radar-sim/wsl_sim.sh
set -euo pipefail

PROJECT_DIR="/mnt/e/1/radar-sim"
cd "$PROJECT_DIR"

# 屏蔽 conda 干扰
export PATH=/usr/bin:/bin:/opt/ros/humble/bin:/usr/local/bin:$PATH
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONHOME PYTHONPATH 2>/dev/null || true

# 加载 ROS2 环境
source /opt/ros/humble/setup.bash
source "$PROJECT_DIR/install/setup.bash"

echo "========================================"
echo "  radar-sim WSL 仿真环境"
echo "========================================"
echo ""
echo "用法："
echo "  基础仿真:     ros2 launch radar_gimbal_gazebo sim.launch.py"
echo "  带运动无人机:  ros2 launch radar_gimbal_gazebo sim.launch.py drone_random_motion:=true"
echo "  带跟踪:       ros2 launch radar_gimbal_gazebo sim.launch.py enable_lasertracking:=true"
echo ""
echo "云台控制:"
echo "  ros2 run radar_gimbal_gazebo gimbal_terminal_control"
echo ""
echo "启动 rviz2:"
echo "  rviz2"
echo ""

exec "$SHELL"
