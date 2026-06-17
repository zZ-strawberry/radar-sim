cmake_minimum_required(VERSION 3.8)
project(radar_gimbal_gazebo)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch urdf config worlds meshes
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/setup_nvidia_kernel_modules.sh
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/gimbal_terminal_control.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME gimbal_terminal_control
)
install(
  PROGRAMS scripts/drone_random_motion.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME drone_random_motion
)
install(
  PROGRAMS scripts/gimbal_command_slew_limiter.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME gimbal_command_slew_limiter
)
install(
  PROGRAMS scripts/lasertracking_tracker.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME lasertracking_tracker
)
install(
  PROGRAMS scripts/gimbal_comm_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME gimbal_comm_bridge
)

ament_package()
