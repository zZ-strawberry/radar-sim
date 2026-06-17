#!/usr/bin/python3
"""
将上层命令做 ACC/DEC 限坡后转发给硬件控制器。

链路:
  /gimbal_position_controller/commands      (上层原始命令，保持兼容)
      -> 限坡
  /gimbal_position_controller_hw/commands   (实际喂给 ros2_control)

输入支持两种格式:
  1) [yaw, pitch]  (历史兼容，弧度)
  2) [pitch, roll, v_pitch, v_roll]  (对齐下位机协议，弧度/弧度每秒)
     其中 roll 对应仿真 yaw_joint。
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class GimbalCommandSlewLimiter(Node):
    def __init__(self) -> None:
        super().__init__("gimbal_command_slew_limiter")

        self.declare_parameter("input_topic", "/gimbal_position_controller/commands")
        self.declare_parameter("output_topic", "/gimbal_position_controller_hw/commands")
        self.declare_parameter("update_rate_hz", 500.0)

        # 默认使用“快速响应版”限坡，避免位置控制显著滞后
        self.declare_parameter("yaw_acc", 35.0)
        self.declare_parameter("yaw_dec", 45.0)
        self.declare_parameter("yaw_vmax", 30.0)

        self.declare_parameter("pitch_acc", 60.0)
        self.declare_parameter("pitch_dec", 80.0)
        self.declare_parameter("pitch_vmax", 30.0)

        # 目标速度生成增益（位置误差->期望速度），仅用于形成平滑轨迹
        self.declare_parameter("tracking_kp", 20.0)
        self.declare_parameter("snap_epsilon", 1e-4)

        in_topic = str(self.get_parameter("input_topic").value)
        out_topic = str(self.get_parameter("output_topic").value)
        self._rate = float(self.get_parameter("update_rate_hz").value)

        self._yaw_acc = float(self.get_parameter("yaw_acc").value)
        self._yaw_dec = float(self.get_parameter("yaw_dec").value)
        self._yaw_vmax = float(self.get_parameter("yaw_vmax").value)
        self._pitch_acc = float(self.get_parameter("pitch_acc").value)
        self._pitch_dec = float(self.get_parameter("pitch_dec").value)
        self._pitch_vmax = float(self.get_parameter("pitch_vmax").value)
        self._kp = float(self.get_parameter("tracking_kp").value)
        self._snap_eps = float(self.get_parameter("snap_epsilon").value)

        self._target = [0.0, 0.0]  # [yaw, pitch]
        self._ff_vel = [0.0, 0.0]  # [yaw, pitch] 速度前馈
        self._out = [0.0, 0.0]
        self._vel = [0.0, 0.0]

        self._sub = self.create_subscription(Float64MultiArray, in_topic, self._on_cmd, 20)
        self._pub = self.create_publisher(Float64MultiArray, out_topic, 20)
        self._timer = self.create_timer(1.0 / max(1.0, self._rate), self._on_timer)

        self.get_logger().info(
            f"限坡器已启动: {in_topic} -> {out_topic}, rate={self._rate:.1f}Hz"
        )

    def _on_cmd(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 2:
            self.get_logger().warn("收到命令长度<2，已忽略")
            return
        # serial_comm 对齐模式: [pitch, roll, v_pitch, v_roll]
        if len(msg.data) >= 4:
            pitch = float(msg.data[0])
            roll = float(msg.data[1])
            v_pitch = float(msg.data[2])
            v_roll = float(msg.data[3])
            # 仿真关节顺序为 [yaw, pitch]，roll -> yaw
            self._target[0] = roll
            self._target[1] = pitch
            self._ff_vel[0] = v_roll
            self._ff_vel[1] = v_pitch
            return

        # 历史兼容: [yaw, pitch]
        self._target[0] = float(msg.data[0])
        self._target[1] = float(msg.data[1])
        self._ff_vel[0] = 0.0
        self._ff_vel[1] = 0.0

    @staticmethod
    def _step_axis(
        out_pos: float,
        out_vel: float,
        target_pos: float,
        acc: float,
        dec: float,
        vmax: float,
        kp: float,
        vel_ff: float,
        dt: float,
        snap_eps: float,
    ) -> tuple[float, float]:
        err = target_pos - out_pos
        if abs(err) <= snap_eps and abs(out_vel) <= snap_eps:
            return target_pos, 0.0

        desired_vel = _clamp(kp * err + vel_ff, -vmax, vmax)
        dv = desired_vel - out_vel

        # 变快用 ACC，变慢/反向用 DEC
        if math.copysign(1.0, desired_vel) != math.copysign(1.0, out_vel) and abs(out_vel) > snap_eps:
            max_dv = dec * dt
        elif abs(desired_vel) > abs(out_vel):
            max_dv = acc * dt
        else:
            max_dv = dec * dt

        dv = _clamp(dv, -max_dv, max_dv)
        new_vel = _clamp(out_vel + dv, -vmax, vmax)
        new_pos = out_pos + new_vel * dt

        # 穿越目标点时直接吸附，避免抖动
        if (target_pos - out_pos) * (target_pos - new_pos) < 0.0:
            return target_pos, 0.0
        return new_pos, new_vel

    def _on_timer(self) -> None:
        dt = 1.0 / max(1.0, self._rate)

        yaw_pos, yaw_vel = self._step_axis(
            self._out[0],
            self._vel[0],
            self._target[0],
            self._yaw_acc,
            self._yaw_dec,
            self._yaw_vmax,
            self._kp,
            self._ff_vel[0],
            dt,
            self._snap_eps,
        )
        pitch_pos, pitch_vel = self._step_axis(
            self._out[1],
            self._vel[1],
            self._target[1],
            self._pitch_acc,
            self._pitch_dec,
            self._pitch_vmax,
            self._kp,
            self._ff_vel[1],
            dt,
            self._snap_eps,
        )

        self._out[0], self._vel[0] = yaw_pos, yaw_vel
        self._out[1], self._vel[1] = pitch_pos, pitch_vel

        msg = Float64MultiArray()
        msg.data = [self._out[0], self._out[1]]
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = GimbalCommandSlewLimiter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

