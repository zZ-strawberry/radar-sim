#!/usr/bin/python3
"""
在终端向 ros2_control 发送云台角度指令（对齐下位机四元组协议）。

使用 /usr/bin/python3：ROS Humble 的 rclpy 针对系统 Python 3.10；若 shebang 为 env python3，
conda base 可能指向 3.12 导致 _rclpy_pybind11 加载失败。

当前 sim.launch.py 只 spawn 一台雷达（实体名 radar_gimbal），可动云台为两轴：
  - pitch_joint 俯仰（弧度，约 ±1.2）
  - yaw_joint   对应下位机 roll（弧度，约 ±π）

若 URDF 里为 gazebo_ros2_control 配置了 <ros><namespace>/foo</namespace>，用 --ns1 / --ns2 指定
该命名空间（不要带尾斜杠）。默认对应本仓库 sim.launch：根命名空间，话题为
/gimbal_position_controller/commands。

用法示例:
  ros2 run radar_gimbal_gazebo gimbal_terminal_control
  ros2 run radar_gimbal_gazebo gimbal_terminal_control
  ros2 run radar_gimbal_gazebo gimbal_terminal_control -- --rad
  ros2 run radar_gimbal_gazebo gimbal_terminal_control -- --legacy

不确定话题名时: ros2 topic list | grep gimbal_position
"""

from __future__ import annotations

import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def _cmd_topic(ros_ns: str) -> str:
    """ros_ns 为 gazebo_ros2_control 的 <ros><namespace>，空字符串表示根命名空间。"""
    ns = ros_ns.strip().strip("/")
    if not ns:
        return "/gimbal_position_controller/commands"
    return f"/{ns}/gimbal_position_controller/commands"


class GimbalTerminalControl(Node):
    def __init__(self, ros_namespace: str) -> None:
        super().__init__("gimbal_terminal_control")
        t = _cmd_topic(ros_namespace)
        self._pub = self.create_publisher(Float64MultiArray, t, 10)
        self._ns_name = ros_namespace or "(根)"
        self.get_logger().info(
            f"发布: {t}  (serial格式: [pitch, roll, v_pitch, v_roll], roll->yaw_joint)"
        )

    def publish_serial_style(self, pitch: float, roll: float, v_pitch: float, v_roll: float) -> None:
        msg1 = Float64MultiArray()
        msg1.data = [float(pitch), float(roll), float(v_pitch), float(v_roll)]
        self._pub.publish(msg1)

    def publish_legacy(self, yaw: float, pitch: float) -> None:
        msg1 = Float64MultiArray()
        msg1.data = [float(yaw), float(pitch)]
        self._pub.publish(msg1)


def _parse_floats(line: str) -> list[float]:
    parts = line.strip().split()
    return [float(x) for x in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="终端控制云台角度（默认: pitch roll v_pitch v_roll）")
    parser.add_argument(
        "--ns1",
        default="",
        help="第一台对应 gazebo_ros2_control 的 ROS 命名空间；默认空=根命名空间（与当前 sim.launch 一致）",
    )
    parser.add_argument(
        "--rad",
        action="store_true",
        help="输入单位为弧度（默认输入单位为度）",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="兼容旧格式：输入/发送 [yaw, pitch] 两个角度",
    )
    args, _unknown = parser.parse_known_args()

    rclpy.init()
    node = GimbalTerminalControl(args.ns1.strip())

    scale = 1.0 if args.rad else math.pi / 180.0

    print()
    if args.legacy:
        print("兼容模式: 输入两个数 <yaw> <pitch>，单位: " + ("弧度" if args.rad else "度"))
        print("例: 0.3 -0.2   或   q 退出")
    else:
        print("输入四个数: <pitch> <roll> <v_pitch> <v_roll>，单位: " + ("弧度" if args.rad else "度"))
        print("例: 10 -5 30 -20   或   q 退出")
    print()

    try:
        while rclpy.ok():
            try:
                line = input("gimbal> ").strip()
            except EOFError:
                break
            if not line or line.lower() in ("q", "quit", "exit"):
                break
            try:
                vals = _parse_floats(line)
            except ValueError:
                print("解析失败，请输入数字。", file=sys.stderr)
                continue

            if args.legacy:
                if len(vals) != 2:
                    print("需要 2 个数: yaw pitch", file=sys.stderr)
                    continue
                y, p = vals[0] * scale, vals[1] * scale
                node.publish_legacy(y, p)
                node.get_logger().info(f"[{node._ns_name}] yaw={y:.4f} pitch={p:.4f} rad")
            else:
                if len(vals) != 4:
                    print("需要 4 个数: pitch roll v_pitch v_roll", file=sys.stderr)
                    continue
                p, r, vp, vr = [v * scale for v in vals]
                node.publish_serial_style(p, r, vp, vr)
                node.get_logger().info(
                    f"[{node._ns_name}] pitch={p:.4f} roll={r:.4f} v_pitch={vp:.4f} v_roll={vr:.4f} (rad / rad/s)"
                )
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
