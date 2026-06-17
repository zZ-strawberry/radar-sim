#!/usr/bin/env bash
# 实机一键：hw_sdk（海康驱动 + lasertracking_tracker）
# 用法：可把本脚本复制到 ~/start.sh；若不在仓库内，会默认使用 ${HOME}/radar/radar-sim
# 也可：export RADAR_SIM_WS=/你的/雷达仓库路径
# 可传 launch 参数：./start.sh start_wide_camera:=true
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${RADAR_SIM_WS:-}" && -f "${RADAR_SIM_WS}/install/setup.bash" ]]; then
  WS="${RADAR_SIM_WS}"
elif [[ -f "${SCRIPT_DIR}/install/setup.bash" ]]; then
  WS="${SCRIPT_DIR}"
elif [[ -f "${HOME}/radar/radar-sim/install/setup.bash" ]]; then
  WS="${HOME}/radar/radar-sim"
else
  echo "未找到 radar-sim 的 install/setup.bash。" >&2
  echo "请将 start.sh 放在仓库根目录，或保证存在 ${HOME}/radar/radar-sim/install/setup.bash，或设置 RADAR_SIM_WS。" >&2
  exit 1
fi
cd "$WS"

# 弱化 conda 对 ROS2 / 系统 Python 的影响（与 run_sim.sh 一致）
export PATH=/usr/bin:/bin:/opt/ros/humble/bin:/usr/local/bin:$PATH
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONHOME PYTHONPATH 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  conda deactivate 2>/dev/null || true
fi

set +u
source /opt/ros/humble/setup.bash
# 海康 ROS2 驱动在单独工作区时必须 source（见 README）
if [[ -f "${HOME}/hik_ros2_ws/install/setup.bash" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/hik_ros2_ws/install/setup.bash"
fi
# shellcheck source=/dev/null
source "${WS}/install/setup.bash"
set -u

exec ros2 launch radar_gimbal_gazebo hw_sdk.launch.py "$@"
