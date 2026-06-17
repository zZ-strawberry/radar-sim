#!/usr/bin/env bash
set -euo pipefail

cd /home/yy/radar/radar-sim

# 强制使用系统 Python，避免 conda 污染 ROS2 运行时
export PATH=/usr/bin:/bin:/opt/ros/humble/bin:$PATH
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONHOME PYTHONPATH

# 可选：清理上一次编译产物（需要时手动开启）
# rm -rf build install log

# 清理残留实例，避免端口和实体重名冲突
pkill -9 -f gzserver || true
pkill -9 -f gzclient || true
pkill -9 -f spawn_entity.py || true
pkill -9 -f robot_state_publisher || true
pkill -9 -f controller_manager/spawner || true

set +u
source /opt/ros/humble/setup.bash
set -u

echo "[run_sim] colcon build radar_gimbal_gazebo..."
colcon build --packages-select radar_gimbal_gazebo

set +u
source /home/yy/radar/radar-sim/install/setup.bash
set -u

exec ros2 launch radar_gimbal_gazebo sim.launch.py
