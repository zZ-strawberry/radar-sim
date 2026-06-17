#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "serial/serial.h"
#include "tools/thread_safe_queue.hpp"

namespace io
{
// ---------------------------------------------------------------------------
// 跟踪系统 ↔ 云台下位机：分帧协议（与 lasertracking/serial_comm.py 模式3 一致）
// ---------------------------------------------------------------------------
// 帧长 23 字节，小端 float / uint32（与 host 为 LE 的 Python struct.pack('<ffffI') 一致）
// byte0       : 0xCD
// byte1..20   : pitch, roll, v_pitch, v_roll, timestamp_ms（各 float32 + uint32）
// byte21      : CRC8(payload)，poly=0x07, init=0x00，覆盖 byte1..20
// byte22      : 0xDC
// 上行：下位机用同样格式回传姿态（pitch/roll/pitch_rate/roll_rate/timestamp）
inline constexpr uint8_t kTrackingFrameHead = 0xCD;
inline constexpr uint8_t kTrackingFrameTail = 0xDC;
inline constexpr std::size_t kTrackingPayloadBytes = 20;
inline constexpr std::size_t kTrackingFrameBytes = 23;

/// 分帧 payload（wire 顺序与 serial_comm.FRAME_PAYLOAD_FMT = '<ffffI' 一致）
struct __attribute__((packed)) TrackingFramedPayload
{
  float pitch_deg;       // 度，>0 抬头
  float roll_deg;        // 度，>0 左转（与 Python 注释一致）
  float v_pitch_dps;     // 度/秒
  float v_roll_dps;      // 度/秒
  uint32_t timestamp_ms; // 毫秒，uint32 回绕与 Python & 0xFFFFFFFF 一致
};

static_assert(sizeof(TrackingFramedPayload) == kTrackingPayloadBytes);

/// 仅选择物理端口（UART 或 USB 虚拟串口），不改变帧格式
enum class GimbalCommLink : uint8_t
{
  SERIAL = 0,
  USB = 1,
};

/// 下位机最新反馈（由上行帧解析）
struct GimbalState
{
  float pitch_deg;
  float roll_deg;
  float v_pitch_dps;
  float v_roll_dps;
  uint32_t timestamp_ms;
};

/// 跟踪 → 下位机一帧控制量；timestamp_ms==0 表示由 Gimbal::send 自动填当前时间戳
struct TrackingDownCommand
{
  float pitch_deg;
  float roll_deg;
  float v_pitch_dps;
  float v_roll_dps;
  uint32_t timestamp_ms = 0;
};

class Gimbal
{
public:
  Gimbal(const std::string & config_path);

  ~Gimbal();

  GimbalState state() const;
  GimbalCommLink comm_link() const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  /// 与 serial_comm 分帧一致；timestamp_ms==0 时使用当前墙钟毫秒（与 Python time.time()*1000 语义对齐）
  void send(const TrackingDownCommand & cmd);

  /// 便捷：pitch/roll 度，角速度度/秒；内部自动时间戳
  void send(float pitch_deg, float roll_deg, float v_pitch_dps, float v_roll_dps);

  /// 兼容旧调用习惯：yaw→roll、pitch→pitch，水平/垂直角速度→v_roll / v_pitch；control==false 时发全零
  void send(
    bool control, bool fire, float yaw_deg, float yaw_vel_dps, float yaw_acc_unused, float pitch_deg,
    float pitch_vel_dps, float pitch_acc_unused);

private:
  serial::Serial serial_;
  GimbalCommLink comm_link_ = GimbalCommLink::SERIAL;

  std::thread thread_;
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;

  GimbalState state_{};
  tools::ThreadSafeQueue<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
    queue_{1000};

  std::vector<uint8_t> rx_buffer_;

  bool read_some(uint8_t * buffer, size_t max_len, size_t & out_n);
  void read_thread();
  void reconnect();
  bool try_decode_one_frame(TrackingFramedPayload & out);
};

}  // namespace io

#endif  // IO__GIMBAL_HPP
