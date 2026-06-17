#!/usr/bin/python3
from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class GimbalCommBridge(Node):
    """上位机/下位机通信桥:
    - 上位机命令: /lasertracking/upper_cmd
    - 下位机执行入口: /gimbal_position_controller/commands
    - 下位机反馈: /lasertracking/lower_feedback
    """

    def __init__(self) -> None:
        super().__init__("gimbal_comm_bridge")
        self.declare_parameter("upper_cmd_topic", "/lasertracking/upper_cmd")
        self.declare_parameter("lower_cmd_topic", "/gimbal_position_controller/commands")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("lower_feedback_topic", "/lasertracking/lower_feedback")

        upper_topic = str(self.get_parameter("upper_cmd_topic").value)
        lower_topic = str(self.get_parameter("lower_cmd_topic").value)
        joint_topic = str(self.get_parameter("joint_state_topic").value)
        feedback_topic = str(self.get_parameter("lower_feedback_topic").value)

        self._lower_cmd_pub = self.create_publisher(Float64MultiArray, lower_topic, 20)
        self._feedback_pub = self.create_publisher(Float64MultiArray, feedback_topic, 20)
        self.create_subscription(Float64MultiArray, upper_topic, self._on_upper_cmd, 20)
        self.create_subscription(JointState, joint_topic, self._on_joint_state, 50)
        self.get_logger().info(
            f"gimbal_comm_bridge 启动: {upper_topic} -> {lower_topic}, feedback={feedback_topic}"
        )

    def _on_upper_cmd(self, msg: Float64MultiArray) -> None:
        # 直接透传到云台控制器入口话题（与下位机协议保持一致）
        self._lower_cmd_pub.publish(msg)

    def _on_joint_state(self, msg: JointState) -> None:
        yaw = 0.0
        pitch = 0.0
        for idx, name in enumerate(msg.name):
            if idx >= len(msg.position):
                continue
            if name == "yaw_joint":
                yaw = float(msg.position[idx])
            elif name == "pitch_joint":
                pitch = float(msg.position[idx])

        fb = Float64MultiArray()
        # 反馈格式: [pitch, roll(yaw), v_pitch, v_roll]
        fb.data = [pitch, yaw, 0.0, 0.0]
        self._feedback_pub.publish(fb)


def main() -> None:
    rclpy.init()
    node = GimbalCommBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
