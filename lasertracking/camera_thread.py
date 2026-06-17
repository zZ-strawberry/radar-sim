import threading
import time


class ThreadedCamera:
    """线程化相机包装器 - 独立线程持续采集最新帧。"""

    def __init__(self, camera, enable_thread=True):
        self.camera = camera
        self.enable_thread = bool(enable_thread)
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.stopped = True
        self.frame_count = 0
        self.thread = None

        if self.enable_thread:
            self._start_thread(wait_first_frame=True)
        else:
            print("✓ 相机运行在同步模式（主循环直接调用read）")

    def _start_thread(self, wait_first_frame=False):
        if self.thread is not None and self.thread.is_alive():
            return
        self.stopped = False
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        if wait_first_frame:
            print("⏳ 等待相机采集第一帧...", end='', flush=True)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                with self.lock:
                    if self.frame is not None:
                        break
                time.sleep(0.05)
            else:
                print(" 超时! 请检查相机连接")
            print(" OK")
        print("✓ 相机独立线程已启动（异步采集最新帧）")

    def _stop_thread(self):
        if self.thread is None:
            return
        self.stopped = True
        self.thread.join(timeout=2.0)
        self.thread = None
        print("相机采集线程已停止")

    def set_thread_mode(self, enable_thread):
        """运行时切换异步/同步采集模式。"""
        enable_thread = bool(enable_thread)
        if enable_thread == self.enable_thread:
            return
        self.enable_thread = enable_thread
        if self.enable_thread:
            self._start_thread(wait_first_frame=False)
        else:
            self._stop_thread()
            print("✓ 相机独立线程已禁用（切换到同步模式）")

    def _update_loop(self):
        while not self.stopped:
            try:
                ret, frame = self.camera.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                        self.ret = True
                        self.frame_count += 1
            except Exception as e:
                print(f"⚠️ 相机线程读取异常: {e}")
                time.sleep(0.01)

    def read(self):
        if not self.enable_thread:
            return self.camera.read()

        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def get_frame_count(self):
        if not self.enable_thread:
            return 0
        with self.lock:
            return self.frame_count

    def release(self):
        if self.enable_thread:
            self._stop_thread()
        self.camera.release()

    def set_camera_params(self, exposure_time=None, gain=None):
        return self.camera.set_camera_params(exposure_time, gain)

    def get_camera_params(self):
        return self.camera.get_camera_params()

    @property
    def exposure_time(self):
        return self.camera.exposure_time

    @property
    def gain(self):
        return self.camera.gain
