## 启动仿真
```bash
#不带运动无人机的仿真
ros2 launch radar_gimbal_gazebo sim.launch.py
#带运动无人机的仿真
ros2 launch radar_gimbal_gazebo sim.launch.py drone_random_motion:=true 
```

## 运行云台控制
```bash
# 新协议
# pitch roll v_pitch v_roll
# 单位默认是度（角速度为度/秒）
ros2 run radar_gimbal_gazebo gimbal_terminal_control

# 输入单位改为弧度/弧度每秒
ros2 run radar_gimbal_gazebo gimbal_terminal_control -- --rad

# 兼容旧协议（两个数据）：yaw pitch
ros2 run radar_gimbal_gazebo gimbal_terminal_control -- --legacy
```

## rviz2查看相机话题
```bash
rviz2
```

## lasertracking 仿真链路（双相机检测 + 长焦跟踪 + 上下位机通信）
```bash
# 1) 安装依赖（仅首次）
/usr/bin/python3 -m pip install ultralytics

# 2) 启动仿真并启用跟踪（MODEL替换为你的无人机检测模型）
ros2 launch radar_gimbal_gazebo sim.launch.py \
  enable_lasertracking:=true 
```

## lasertracking 实机运行
```bash
# 0) 进入工作区并加载环境
cd /home/yy/radar/radar-sim
source /opt/ros/humble/setup.bash
source install/setup.bash
# 海康 ROS2 驱动（hik_camera_ros2_driver）若装在单独工作区，必须再 source，否则 launch 会报 package not found
source ~/hik_ros2_ws/install/setup.bash

# 1) 仅启动跟踪节点（相机驱动已单独启动时用）
ros2 launch radar_gimbal_gazebo hw.launch.py

# 2) 启动相机驱动 + 跟踪节点（一体启动，推荐）
ros2 launch radar_gimbal_gazebo hw_sdk.launch.py

# 3) 若只想启动跟踪，不拉起相机驱动（仍使用 hw_sdk.launch.py）
ros2 launch radar_gimbal_gazebo hw_sdk.launch.py start_cameras:=false

# 4) 仅启动长焦海康节点、不启跟踪（boresight 用 --ros-image-topic 取图时）
#    依赖已安装 hik_camera_ros2_driver；短焦可关 start_wide_camera:=false
ros2 launch radar_gimbal_gazebo hw_sdk.launch.py \
  start_tracker:=false start_wide_camera:=false
```

## 激光校准（boresight）运行命令
```bash
# 0) 进入工作区并加载环境
cd /home/yy/radar/radar-sim
source /opt/ros/humble/setup.bash
source install/setup.bash
source ~/hik_ros2_ws/install/setup.bash

# 1) 先检查配置与路径（不打开相机）
python3 lasertracking/calibration/boresight_calibrator.py \
  --config lasertracking/config_tracking.yaml \
  --check-config

# 2) 标准校准（带检测器）
python3 lasertracking/calibration/boresight_calibrator.py \
  --config lasertracking/config_tracking.yaml

# 3) 手动校准（不加载检测器）
python3 lasertracking/calibration/boresight_calibrator.py \
  --config lasertracking/config_tracking.yaml \
  --no-detector

# 3b) 用 ROS 长焦图标定（终端 A 先起「仅长焦海康」，见上文实机 4)）
python3 lasertracking/calibration/boresight_calibrator.py \
  --config lasertracking/config_tracking.yaml \
  --no-detector \
  --ros-image-topic /gimbal_camera/image_raw

# 4) 退出时默认会自动保存并回写配置；
#    如需关闭“退出自动保存”，加上：
python3 lasertracking/calibration/boresight_calibrator.py \
  --config lasertracking/config_tracking.yaml \
  --no-auto-save-on-exit
```

## 模型.pt转.engine
```bash
/usr/bin/python3 -c "
from ultralytics import YOLO
YOLO('best_new.pt', task='detect').export(
 format='engine',
 imgsz=640,
 half=True,
 device=0
)
"
```

## 检查设备
```bash
for d in /sys/bus/usb/devices/*; do   [[ "$(cat "$d/idVendor" 2>/dev/null)" == "2bdf" ]] || continue;   echo "serial=$(cat "$d/serial")  product=$(cat "$d/product")"; done
```