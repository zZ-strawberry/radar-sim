#include "gimbal.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
namespace
{
constexpr double kDegToRad = 3.14159265358979323846 / 180.0;

GimbalCommLink parse_comm_link(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (s == "usb") return GimbalCommLink::USB;
  return GimbalCommLink::SERIAL;
}

/// 与 serial_comm.SerialCommunicator._crc8 一致：poly=0x07, init=0x00
uint8_t crc8_payload(const uint8_t * payload, std::size_t len)
{
  uint8_t crc = 0x00;
  for (std::size_t i = 0; i < len; ++i) {
    crc ^= static_cast<uint8_t>(payload[i]);
    for (int b = 0; b < 8; ++b) {
      if (crc & 0x80)
        crc = static_cast<uint8_t>((static_cast<uint16_t>(crc) << 1u) ^ 0x07u);
      else
        crc = static_cast<uint8_t>(static_cast<uint16_t>(crc) << 1u);
    }
  }
  return crc;
}

uint32_t wall_clock_ms_u32()
{
  using namespace std::chrono;
  const auto ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  return static_cast<uint32_t>(static_cast<uint64_t>(ms) & 0xffffffffu);
}

void pack_framed(uint8_t out[kTrackingFrameBytes], float pitch_deg, float roll_deg, float v_pitch_dps, float v_roll_dps, uint32_t timestamp_ms)
{
  out[0] = kTrackingFrameHead;
  TrackingFramedPayload pl{pitch_deg, roll_deg, v_pitch_dps, v_roll_dps, timestamp_ms};
  std::memcpy(out + 1, &pl, sizeof(pl));
  out[21] = crc8_payload(out + 1, kTrackingPayloadBytes);
  out[22] = kTrackingFrameTail;
}

Eigen::Quaterniond quat_from_pitch_roll_deg(float pitch_deg, float roll_deg)
{
  const double p = static_cast<double>(pitch_deg) * kDegToRad;
  const double r = static_cast<double>(roll_deg) * kDegToRad;
  return Eigen::Quaterniond(
    Eigen::AngleAxisd(r, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(p, Eigen::Vector3d::UnitY()));
}
}  // namespace

Gimbal::Gimbal(const std::string & config_path)
{
  auto yaml = tools::load(config_path);

  std::string link_str = "serial";
  try {
    link_str = tools::read<std::string>(yaml, "comm_link");
  } catch (...) {
  }
  comm_link_ = parse_comm_link(link_str);

  std::string port;
  if (comm_link_ == GimbalCommLink::USB) {
    std::string usb_port;
    try {
      usb_port = tools::read<std::string>(yaml, "usb_port");
    } catch (...) {
    }
    port = usb_port.empty() ? tools::read<std::string>(yaml, "com_port") : usb_port;
  } else {
    port = tools::read<std::string>(yaml, "com_port");
  }

  try {
    serial_.setPort(port);
    try {
      const int baud = tools::read<int>(yaml, "baud_rate");
      serial_.setBaudrate(static_cast<uint32_t>(baud));
    } catch (...) {
    }
    serial_.open();
  } catch (const std::exception & e) {
    tools::logger()->error(
      "[Gimbal] Failed to open {}: {}", comm_link_ == GimbalCommLink::USB ? "usb" : "serial", e.what());
    exit(1);
  }

  tools::logger()->info(
    "[Gimbal] Opened {} on {} @ {} (tracking framed {} bytes)", comm_link_ == GimbalCommLink::USB ? "USB(CDC)" : "serial",
    port, serial_.getBaudrate(), kTrackingFrameBytes);

  thread_ = std::thread(&Gimbal::read_thread, this);

  queue_.pop();
  tools::logger()->info("[Gimbal] First attitude frame received.");
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

GimbalCommLink Gimbal::comm_link() const
{
  return comm_link_;
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  while (true) {
    auto [q_a, t_a] = queue_.pop();
    auto [q_b, t_b] = queue_.front();
    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t);
    auto k = t_ac / t_ab;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
    if (t < t_a) return q_c;
    if (!(t_a < t && t <= t_b)) continue;

    return q_c;
  }
}

void Gimbal::send(const TrackingDownCommand & cmd)
{
  uint32_t ts = cmd.timestamp_ms;
  if (ts == 0) ts = wall_clock_ms_u32();
  uint8_t frame[kTrackingFrameBytes];
  pack_framed(frame, cmd.pitch_deg, cmd.roll_deg, cmd.v_pitch_dps, cmd.v_roll_dps, ts);
  try {
    serial_.write(frame, sizeof(frame));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write: {}", e.what());
  }
}

void Gimbal::send(float pitch_deg, float roll_deg, float v_pitch_dps, float v_roll_dps)
{
  send(TrackingDownCommand{pitch_deg, roll_deg, v_pitch_dps, v_roll_dps, 0});
}

void Gimbal::send(
  bool control, bool fire, float yaw_deg, float yaw_vel_dps, float yaw_acc_unused, float pitch_deg,
  float pitch_vel_dps, float pitch_acc_unused)
{
  (void)fire;
  (void)yaw_acc_unused;
  (void)pitch_acc_unused;
  if (!control) {
    send(TrackingDownCommand{0.f, 0.f, 0.f, 0.f, 0});
    return;
  }
  send(TrackingDownCommand{pitch_deg, yaw_deg, pitch_vel_dps, yaw_vel_dps, 0});
}

bool Gimbal::read_some(uint8_t * buffer, size_t max_len, size_t & out_n)
{
  out_n = 0;
  try {
    const size_t avail = serial_.available();
    size_t want = (avail > 0) ? std::min(avail, max_len) : 1;
    if (want < 1) want = 1;
    const size_t n = serial_.read(buffer, want);
    out_n = n;
    return n > 0;
  } catch (const std::exception & e) {
    (void)e;
    return false;
  }
}

bool Gimbal::try_decode_one_frame(TrackingFramedPayload & out)
{
  while (rx_buffer_.size() >= kTrackingFrameBytes) {
    if (rx_buffer_[0] != kTrackingFrameHead) {
      const auto it = std::find(rx_buffer_.begin(), rx_buffer_.end(), kTrackingFrameHead);
      if (it == rx_buffer_.end()) {
        rx_buffer_.clear();
        return false;
      }
      rx_buffer_.erase(rx_buffer_.begin(), it);
      continue;
    }

    const uint8_t * frame = rx_buffer_.data();
    if (frame[kTrackingFrameBytes - 1] != kTrackingFrameTail) {
      rx_buffer_.erase(rx_buffer_.begin());
      continue;
    }

    const uint8_t got_crc = frame[kTrackingFrameBytes - 2];
    if (got_crc != crc8_payload(&frame[1], kTrackingPayloadBytes)) {
      tools::logger()->debug("[Gimbal] CRC8 mismatch, resync.");
      rx_buffer_.erase(rx_buffer_.begin());
      continue;
    }

    std::memcpy(&out, &frame[1], sizeof(out));
    rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + static_cast<std::ptrdiff_t>(kTrackingFrameBytes));
    return true;
  }
  return false;
}

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started (tracking framed protocol).");
  int error_count = 0;

  while (!quit_) {
    if (error_count > 5000) {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Too many read errors, reconnecting...");
      reconnect();
      continue;
    }

    uint8_t chunk[256];
    size_t n = 0;
    if (!read_some(chunk, sizeof(chunk), n)) {
      error_count++;
      continue;
    }
    error_count = 0;

    std::lock_guard<std::mutex> lock(mutex_);
    if (rx_buffer_.size() + n > 8192) {
      tools::logger()->warn("[Gimbal] RX buffer overflow, clearing.");
      rx_buffer_.clear();
    }
    rx_buffer_.insert(rx_buffer_.end(), chunk, chunk + n);

    TrackingFramedPayload pl{};
    while (try_decode_one_frame(pl)) {
      const auto t = std::chrono::steady_clock::now();
      const Eigen::Quaterniond qg = quat_from_pitch_roll_deg(pl.pitch_deg, pl.roll_deg);
      queue_.push({qg, t});

      state_.pitch_deg = pl.pitch_deg;
      state_.roll_deg = pl.roll_deg;
      state_.v_pitch_dps = pl.v_pitch_dps;
      state_.v_roll_dps = pl.v_roll_dps;
      state_.timestamp_ms = pl.timestamp_ms;
    }
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  const int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    tools::logger()->warn(
      "[Gimbal] Reconnecting {}, attempt {}/{}...",
      comm_link_ == GimbalCommLink::USB ? "USB(CDC)" : "serial", i + 1, max_retry_count);
    try {
      serial_.close();
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (...) {
    }

    try {
      serial_.open();
      {
        std::lock_guard<std::mutex> lock(mutex_);
        rx_buffer_.clear();
      }
      queue_.clear();
      tools::logger()->info(
        "[Gimbal] Reconnected {} successfully.", comm_link_ == GimbalCommLink::USB ? "USB(CDC)" : "serial");
      break;
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

}  // namespace io
