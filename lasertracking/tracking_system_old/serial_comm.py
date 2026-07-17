"""串口通信模块，自动适配有/无串口模式。

协议:
  模式2 raw (16B LE): struct.pack('<ffff', pitch, roll, v_pitch, v_roll)
  模式3 framed (23B LE): 0xCD + payload(20B) + CRC8 + 0xDC
    payload = struct.pack('<ffffI', pitch, roll, v_pitch, v_roll, timestamp_ms)
    CRC-8/ATM poly=0x07 init=0x00

角度: pitch>0 上仰, roll>0 左转。绝对角，相对上电初始姿态。
"""
import struct
import time

FRAME_HEAD = 0xCD
FRAME_TAIL = 0xDC
FRAME_PAYLOAD_FMT = '<ffffI'
FRAME_PAYLOAD_SIZE = struct.calcsize(FRAME_PAYLOAD_FMT)  # 20
FRAME_SIZE = 1 + FRAME_PAYLOAD_SIZE + 1 + 1  # 23

# 避免与项目中的serial.py冲突
try:
    import sys
    # 临时移除可能冲突的路径
    original_path = sys.path.copy()
    sys.path = [p for p in sys.path if 'RM_serial' not in p]
    
    from serial import Serial as PySerial
    
    # 恢复路径
    sys.path = original_path
    
    SERIAL_AVAILABLE = True
    print(f" PySerial 模块加载成功")
except Exception as e:
    SERIAL_AVAILABLE = False
    PySerial = None
    print(f" PySerial 模块加载失败: {e}")
    print("  将使用调试模式（仅打印数据）")


class SerialCommunicator:
    """串口通信类 - 自动适配有/无串口模式"""
    
    def __init__(self, port='COM8', baudrate=115200, timeout=0.1,
                 debug_mode=False, raw_payload=True,
                 protocol_mode=None, enable_feedback=False):
        """
        初始化串口通信

        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_obj = None
        self.debug_mode = debug_mode
        # raw_payload: True 时使用裸 payload（16字节，位置+速度），False 时使用分帧协议
        self.raw_payload = raw_payload
        if protocol_mode is None:
            protocol_mode = 'raw' if raw_payload else 'framed'
        self.protocol_mode = str(protocol_mode).lower()
        self.enable_feedback = bool(enable_feedback)
        self.rx_buffer = bytearray()
        self.latest_state = None
        self.tx_seq = 0
        self.connected = False
        
        if not debug_mode and SERIAL_AVAILABLE:
            self._try_connect()
        else:
            print(f"[串口调试模式] 端口={port}, 波特率={baudrate}")

    @staticmethod
    def _crc8(payload_bytes):
        """CRC-8/ATM: poly=0x07, init=0x00, refin/refout=false, xorout=0x00"""
        crc = 0x00
        for b in payload_bytes:
            crc ^= int(b) & 0xFF
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ 0x07) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc

    @classmethod
    def _pack_framed(cls, pitch, roll, v_pitch, v_roll, timestamp_ms):
        payload = struct.pack(
            FRAME_PAYLOAD_FMT,
            float(pitch), float(roll),
            float(v_pitch), float(v_roll),
            int(timestamp_ms) & 0xFFFFFFFF
        )
        chk = cls._crc8(payload)
        return bytes([FRAME_HEAD]) + payload + bytes([chk, FRAME_TAIL])

    @classmethod
    def _unpack_framed(cls, frame_bytes):
        if len(frame_bytes) != FRAME_SIZE:
            return None
        if frame_bytes[0] != FRAME_HEAD or frame_bytes[-1] != FRAME_TAIL:
            return None
        payload = frame_bytes[1:1 + FRAME_PAYLOAD_SIZE]
        checksum = frame_bytes[1 + FRAME_PAYLOAD_SIZE]
        if checksum != cls._crc8(payload):
            return None
        pitch, roll, v_pitch, v_roll, timestamp_ms = struct.unpack(FRAME_PAYLOAD_FMT, payload)
        return {
            'pitch': float(pitch),
            'roll': float(roll),
            'pitch_rate': float(v_pitch),
            'roll_rate': float(v_roll),
            'timestamp': int(timestamp_ms),
            'recv_time': time.time()
        }

    def _parse_feedback_buffer(self):
        parsed = 0
        while True:
            if len(self.rx_buffer) < FRAME_SIZE:
                break

            # 对齐帧头
            if self.rx_buffer[0] != FRAME_HEAD:
                head_idx = self.rx_buffer.find(bytes([FRAME_HEAD]))
                if head_idx == -1:
                    self.rx_buffer.clear()
                    break
                del self.rx_buffer[:head_idx]
                if len(self.rx_buffer) < FRAME_SIZE:
                    break

            frame = bytes(self.rx_buffer[:FRAME_SIZE])
            if frame[-1] != FRAME_TAIL:
                del self.rx_buffer[0]
                continue

            state = self._unpack_framed(frame)
            del self.rx_buffer[:FRAME_SIZE]
            if state is None:
                continue

            self.latest_state = state
            parsed += 1
        return parsed
    
    def _try_connect(self):
        """尝试连接串口"""
        try:
            self.serial_obj = PySerial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self.connected = True
            print(f" 串口连接成功: {self.port} @ {self.baudrate}")
        except Exception as e:
            self.connected = False
            self.debug_mode = True
            print(f" 串口连接失败: {e}")
            print(f"  切换到调试模式")
    

    
    def send_pitch_yaw(self, pitch, roll):
        """
        发送云台控制角: pitch/roll（仅位置接口）
        - framed模式: 23字节分帧，速度字段置0
        - raw模式: 16字节裸payload，等价于 struct.pack('<ffff', pitch, roll, 0.0, 0.0)
        """
        try:
            # 转换为浮点数
            pitch_val = float(pitch)
            roll_val = float(roll)

            if self.protocol_mode == 'framed':
                timestamp_ms = int(time.time() * 1000) & 0xFFFFFFFF
                packet = self._pack_framed(pitch_val, roll_val, 0.0, 0.0, timestamp_ms)
            else:
                # raw模式统一使用16字节（位置+零速度）
                packet = struct.pack('<ffff', pitch_val, roll_val, 0.0, 0.0)

            if self.connected and self.serial_obj:
                self.serial_obj.write(packet)
                return True
            else:
                # 调试模式：打印发送的数据
                print(f"[串口发送-{self.protocol_mode}] pitch={pitch_val:.2f}, roll={roll_val:.2f} ({len(packet)} bytes: {packet.hex()})")
                return False
        except Exception as e:
            print(f"[串口错误] 发送失败: {e}")
            return False

    def send_pitch_yaw_velocity(self, pitch, roll, v_pitch, v_roll):
        """
        发送云台控制角和角速度
        格式: 二进制，小端序，四个float32打包，共16字节
        使用: struct.pack('<ffff', pitch, roll, v_pitch, v_roll)
        """
        try:
            # 转换为浮点数
            pitch_val = float(pitch)
            roll_val = float(roll)
            v_pitch_val = float(v_pitch)
            v_roll_val = float(v_roll)

            if self.protocol_mode == 'framed':
                timestamp_ms = int(time.time() * 1000) & 0xFFFFFFFF
                packet = self._pack_framed(pitch_val, roll_val, v_pitch_val, v_roll_val, timestamp_ms)
            else:
                # 使用小端字节序打包四个浮点数
                packet = struct.pack('<ffff', pitch_val, roll_val, v_pitch_val, v_roll_val)

            if self.connected and self.serial_obj:
                self.serial_obj.write(packet)
                return True
            else:
                # 调试模式：打印发送的数据
                print(f"[串口发送+速度] pitch={pitch_val:6.2f}° roll={roll_val:6.2f}° "
                      f"v_pitch={v_pitch_val:6.2f}°/s v_roll={v_roll_val:6.2f}°/s "
                      f"({len(packet)} bytes: {packet.hex()})")
                return False
        except Exception as e:
            print(f"[串口错误] 发送失败: {e}")
            return False
    
    def send_raw(self, data):
        """发送原始字节数据"""
        try:
            if self.connected and self.serial_obj:
                self.serial_obj.write(data)
                return True
            else:
                print(f"[串口发送] 原始数据: {data.hex()}")
                return False
        except Exception as e:
            print(f"[串口错误] 发送失败: {e}")
            return False
    
    def read(self, size=1):
        """读取数据"""
        if self.connected and self.serial_obj:
            try:
                return self.serial_obj.read(size)
            except Exception as e:
                print(f"[串口错误] 读取失败: {e}")
                return b''
        return b''

    def poll_feedback(self, max_read=256):
        """轮询并解析回传姿态帧（分帧模式下有效）"""
        if self.protocol_mode != 'framed' or not self.enable_feedback:
            return None
        if not (self.connected and self.serial_obj):
            return self.latest_state

        try:
            available = int(getattr(self.serial_obj, 'in_waiting', 0))
            if available > 0:
                n = min(max_read, available)
                data = self.serial_obj.read(n)
                if data:
                    self.rx_buffer.extend(data)
                    self._parse_feedback_buffer()
        except Exception as e:
            print(f"[串口错误] 回传解析失败: {e}")
            return None

        return self.latest_state

    def get_latest_state(self, max_age_s=None):
        if self.latest_state is None:
            return None
        if max_age_s is None:
            return self.latest_state
        age = time.time() - float(self.latest_state.get('recv_time', 0.0))
        if age <= float(max_age_s):
            return self.latest_state
        return None
    
    def is_open(self):
        """检查串口是否打开"""
        if self.connected and self.serial_obj:
            return self.serial_obj.is_open
        return False
    
    def close(self):
        """关闭串口"""
        if self.serial_obj:
            try:
                self.serial_obj.close()
                print(" 串口已关闭")
            except Exception as e:
                print(f" 关闭串口失败: {e}")
        self.connected = False


# 便捷函数：创建串口通信对象
def create_serial(port='COM8', baudrate=115200, debug_if_fail=True,
                  raw_payload=True, protocol_mode=None, enable_feedback=False):
    #创建串口通信对象
    comm = SerialCommunicator(
        port=port,
        baudrate=baudrate,
        raw_payload=raw_payload,
        protocol_mode=protocol_mode,
        enable_feedback=enable_feedback
    )
    
    if not comm.connected and debug_if_fail:
        print("提示: 使用调试模式，数据将打印到控制台")
    
    return comm


# 测试代码
if __name__ == "__main__":
    print("="*50)
    print("串口通信模块测试")
    print("="*50)
    
    # 创建串口对象（自动适配调试模式）
    comm = create_serial(port='COM8', baudrate=115200)
    
    # 关闭串口
    comm.close()
    
    print("\n测试完成")
