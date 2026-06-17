#!/usr/bin/env python3
"""Gazebo 中模型随机运动：主方向往返 + 可选竖直方向；速度每步带抖动，实现实时变化。"""
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from gazebo_msgs.srv import SetEntityState


def _axis_index(axis: str) -> int:
    a = axis.strip().lower()
    if a == "x":
        return 0
    if a == "y":
        return 1
    if a == "z":
        return 2
    raise ValueError(f"axis 必须是 x/y/z，收到: {axis!r}")


class DroneRandomMotion(Node):
    def __init__(self):
        super().__init__("drone_random_motion")
        self.declare_parameter("model_name", "drone")
        self.declare_parameter("center_x", 20.0)
        self.declare_parameter("center_y", 0.0)
        self.declare_parameter("center_z", 1.0)
        self.declare_parameter("axis", "x")
        self.declare_parameter("half_range_m", 1.5)
        self.declare_parameter("speed_min", 0.2)
        self.declare_parameter("speed_max", 0.8)
        # 每步对 |v| 加减随机量（m/s），再限制在 [speed_min,speed_max]，使速度连续变化
        self.declare_parameter("speed_jitter_mps", 0.12)
        # 相对 center_z 的竖直偏移范围（米）；仅当主运动轴为 x 或 y 时生效；0 表示不上下动
        self.declare_parameter("height_half_range_m", 0.0)
        # <0 表示与 speed_min/max 相同
        self.declare_parameter("height_speed_min", -1.0)
        self.declare_parameter("height_speed_max", -1.0)
        self.declare_parameter("height_speed_jitter_mps", -1.0)
        # 高度方向随机行为：可在到达边界前停住或提前反向
        self.declare_parameter("height_stop_prob", 0.03)
        self.declare_parameter("height_reverse_prob", 0.02)
        self.declare_parameter("height_pause_min_s", 0.15)
        self.declare_parameter("height_pause_max_s", 0.6)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("set_entity_state_service", "/set_entity_state")

        self._model = self.get_parameter("model_name").get_parameter_value().string_value
        self._cx = self.get_parameter("center_x").get_parameter_value().double_value
        self._cy = self.get_parameter("center_y").get_parameter_value().double_value
        self._cz = self.get_parameter("center_z").get_parameter_value().double_value
        self._axis_i = _axis_index(
            self.get_parameter("axis").get_parameter_value().string_value
        )
        self._half = max(
            1e-6, self.get_parameter("half_range_m").get_parameter_value().double_value
        )
        self._v_min = max(
            0.0, self.get_parameter("speed_min").get_parameter_value().double_value
        )
        self._v_max = max(
            self._v_min + 1e-6,
            self.get_parameter("speed_max").get_parameter_value().double_value,
        )
        self._jitter = max(
            0.0, self.get_parameter("speed_jitter_mps").get_parameter_value().double_value
        )
        self._rate = max(1.0, self.get_parameter("publish_rate").get_parameter_value().double_value)
        self._srv_name = (
            self.get_parameter("set_entity_state_service").get_parameter_value().string_value
        )

        self._height_half = max(
            0.0, self.get_parameter("height_half_range_m").get_parameter_value().double_value
        )
        h_min = self.get_parameter("height_speed_min").get_parameter_value().double_value
        h_max = self.get_parameter("height_speed_max").get_parameter_value().double_value
        h_j = self.get_parameter("height_speed_jitter_mps").get_parameter_value().double_value
        self._hv_min = self._v_min if h_min < 0 else max(0.0, h_min)
        self._hv_max = self._v_max if h_max < 0 else max(self._hv_min + 1e-6, h_max)
        self._hjitter = self._jitter if h_j < 0 else max(0.0, h_j)
        self._h_stop_prob = max(
            0.0, min(1.0, self.get_parameter("height_stop_prob").get_parameter_value().double_value)
        )
        self._h_reverse_prob = max(
            0.0, min(1.0, self.get_parameter("height_reverse_prob").get_parameter_value().double_value)
        )
        self._h_pause_min_s = max(
            0.0, self.get_parameter("height_pause_min_s").get_parameter_value().double_value
        )
        self._h_pause_max_s = max(
            self._h_pause_min_s,
            self.get_parameter("height_pause_max_s").get_parameter_value().double_value,
        )

        # 主方向：axis 为 z 时不启用独立高度振荡（与主运动重合）
        self._use_height_axis = self._height_half > 1e-9 and self._axis_i != 2
        if self._height_half > 1e-9 and self._axis_i == 2:
            self.get_logger().info(
                "主运动轴为 z 时已沿竖直方向运动，已忽略 height_half_range_m"
            )

        self._p = 0.0
        self._v = self._new_speed(self._v_min, self._v_max)
        self._p_z = 0.0
        self._v_z = self._new_speed(self._hv_min, self._hv_max)
        self._z_pause_ticks_left = 0

        self._client = self.create_client(SetEntityState, self._srv_name)
        self._timer = self.create_timer(1.0 / self._rate, self._tick)
        self._warned_fail = False
        self._warned_no_service = False
        self._tick_count = 0

    def _new_speed(self, vmin: float, vmax: float) -> float:
        mag = random.uniform(vmin, vmax)
        return mag * random.choice((-1.0, 1.0))

    def _apply_jitter(self, v: float, vmin: float, vmax: float, jitter: float) -> float:
        if jitter <= 0.0:
            return v
        sign = 1.0 if v >= 0 else -1.0
        mag = abs(v) + random.uniform(-jitter, jitter)
        mag = max(vmin, min(vmax, mag))
        return sign * mag

    def _step_axis(self, p: float, v: float, half: float, vmin: float, vmax: float, jitter: float):
        """积分、边界反弹；每步末对速度加抖动，使 |v| 在 [vmin,vmax] 内实时变化。"""
        dt = 1.0 / self._rate
        p += v * dt
        if p > half:
            p = half
            v = -abs(random.uniform(vmin, vmax))
        elif p < -half:
            p = -half
            v = abs(random.uniform(vmin, vmax))
        v = self._apply_jitter(v, vmin, vmax, jitter)
        return p, v

    def _step_height_axis(self):
        """高度轴可随机暂停/随机提前换向，不必等到最高点再下降。"""
        if self._z_pause_ticks_left > 0:
            self._z_pause_ticks_left -= 1
            self._v_z = 0.0
            return

        if random.random() < self._h_stop_prob:
            pause_s = random.uniform(self._h_pause_min_s, self._h_pause_max_s)
            self._z_pause_ticks_left = max(1, int(round(pause_s * self._rate)))
            self._v_z = 0.0
            return

        if random.random() < self._h_reverse_prob:
            sign = -1.0 if self._v_z >= 0.0 else 1.0
            self._v_z = sign * random.uniform(self._hv_min, self._hv_max)

        self._p_z, self._v_z = self._step_axis(
            self._p_z,
            self._v_z,
            self._height_half,
            self._hv_min,
            self._hv_max,
            self._hjitter,
        )

    def _pose_from_offset(self) -> Pose:
        pos = [self._cx, self._cy, self._cz]
        pos[self._axis_i] += self._p
        if self._use_height_axis:
            pos[2] += self._p_z
        return Pose(
            position=Point(x=pos[0], y=pos[1], z=pos[2]),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )

    def _tick(self):
        self._tick_count += 1
        if not self._client.service_is_ready():
            if not self._warned_no_service and self._tick_count >= int(self._rate):
                self.get_logger().warning(
                    f"约 1s 内未等到服务 {self._srv_name!r}：world 需加载插件 "
                    "`libgazebo_ros_state.so`（本仓库 worlds/empty.world 已包含）"
                )
                self._warned_no_service = True
            return

        self._p, self._v = self._step_axis(
            self._p, self._v, self._half, self._v_min, self._v_max, self._jitter
        )
        if self._use_height_axis:
            self._step_height_axis()

        req = SetEntityState.Request()
        req.state.name = self._model
        req.state.reference_frame = "world"
        req.state.pose = self._pose_from_offset()
        req.state.twist = Twist()
        self._client.call_async(req).add_done_callback(self._on_set_done)

    def _on_set_done(self, fut):
        try:
            ok = fut.result().success
        except Exception as e:
            if not self._warned_fail:
                self.get_logger().warning(f"set_entity_state 调用异常: {e}")
                self._warned_fail = True
            return
        if not ok and not self._warned_fail:
            self.get_logger().warning(
                "set_entity_state 返回 success=false（模型是否已生成、名称是否为 drone？）"
            )
            self._warned_fail = True


def main():
    rclpy.init()
    node = DroneRandomMotion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
