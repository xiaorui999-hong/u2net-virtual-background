# -*- coding: utf-8 -*-
"""会议语音的本地收发层：麦克风采集与扬声器混音播放。

网络传输由 lan_conference.py 负责，本模块只管“本机的声音怎么进出”。

【线程模型】
  - 采集回调（PortAudio 线程）：每拿到 20 ms 一块 PCM，原样交给上层发送。
  - 播放回调（PortAudio 线程）：从每个远端说话人的缓冲里各取一块，相加后输出。

【为什么在客户端混音，而不是在主机混音】
  若由主机混音，它必须为每个听众单独准备一份“去掉他自己”的混音，N 个人就是 N 路
  编码；而主机只做转发时，每台客户端把收到的几路 int16 相加几乎不花 CPU。代价是每人
  多占一路带宽——16 kHz 单声道 PCM 只有 256 kbps，在局域网里远比视频便宜。

【为什么用原始 PCM 而不是 Opus】
  Opus 能把带宽压到 24 kbps 左右，但 Windows 上要额外摆平 libopus 的 DLL。当前设计
  里语音只占视频带宽的一小块，先不引入这个原生依赖；将来若带宽吃紧，把 _encode /
  _decode 换成编解码函数即可，协议层不用动。
"""

import threading

import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False


SAMPLE_RATE = 16000
# ↑ 16 kHz 足够覆盖人声频段（300–3400 Hz），又比 44.1 kHz 省一半带宽。
BLOCK_SAMPLES = 320
# ↑ 20 ms 一块：短到听不出延迟，长到不会把包发得太碎。
BLOCK_BYTES = BLOCK_SAMPLES * 2
MAX_BUFFER_BYTES = BLOCK_BYTES * 8
# ↑ 单个说话人最多缓冲 160 ms。超过就丢最旧的一块：宁可让声音断一下，也不能让它越拖越久。
MAX_SENDERS = 64
# ↑ 最多为多少个说话人保留播放缓冲。必须大于会议人数上限，否则排在后半段的人一开口
#   就会被静默丢掉。留出余量是为了容忍“有人刚退出、旧缓冲还没清掉”的重叠时刻。


def list_devices():
    """枚举音频设备，返回 (输入设备, 输出设备)，元素为 (索引, 名称)。"""
    inputs, outputs = [], []
    if not HAS_SOUNDDEVICE:
        return inputs, outputs
    try:
        for index, device in enumerate(sd.query_devices()):
            name = device.get('name', f'设备{index}')
            if device.get('max_input_channels', 0) > 0:
                inputs.append((index, name))
            if device.get('max_output_channels', 0) > 0:
                outputs.append((index, name))
    except Exception as e:
        print(f"枚举音频设备失败: {e}")
    return inputs, outputs


class AudioLink:
    """会议语音的采集与播放。

    典型用法::

        link = AudioLink(on_capture=conference.publish_audio)
        link.start(input_device, output_device)
        link.push_remote(sender_id, pcm)   # 收到远端语音时调用
        link.set_muted(True)
        link.stop()

    设计上“一直能听、自己决定是否说”：start() 之后扬声器持续播放远端语音，麦克风虽然
    已打开，但 muted 为 True 时一块都不会发出去。
    """

    def __init__(self, on_capture):
        self.on_capture = on_capture
        # ↑ 采集到一块 PCM 时调用，参数是 bytes。必须是快函数——它在音频回调线程里执行。

        self.input_stream = None
        self.output_stream = None
        self.running = False
        self.muted = True
        self.volume = 1.0

        self.lock = threading.Lock()
        self.buffers = {}
        # ↑ sender_id -> bytearray，每个远端说话人一份待播放的 PCM。

    # ---------------- 生命周期 ----------------

    def start(self, input_device=None, output_device=None):
        """打开麦克风与扬声器。设备为 None 时使用系统默认设备。"""
        if not HAS_SOUNDDEVICE:
            raise RuntimeError("未安装 sounddevice，无法启用会议语音。请 pip install sounddevice")

        self.stop()
        with self.lock:
            self.buffers.clear()
        self.running = True

        try:
            self.input_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                blocksize=BLOCK_SAMPLES, device=input_device,
                callback=self._capture_callback
            )
            self.input_stream.start()

            self.output_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                blocksize=BLOCK_SAMPLES, device=output_device,
                callback=self._playback_callback
            )
            self.output_stream.start()
        except Exception:
            # 两个设备里任意一个打不开，就把另一个也收回去，避免留下半开的状态。
            self.stop()
            raise

    def stop(self):
        """关闭两个音频流并清空播放缓冲。可重复调用。"""
        self.running = False
        for stream in (self.input_stream, self.output_stream):
            if stream is None:
                continue
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self.input_stream = None
        self.output_stream = None
        with self.lock:
            self.buffers.clear()

    # ---------------- 控制 ----------------

    def set_muted(self, muted):
        """静音只停止发送麦克风，不影响扬声器继续播放他人语音。"""
        self.muted = bool(muted)

    def set_volume(self, volume):
        """播放音量，0.0–1.0。"""
        self.volume = max(0.0, min(1.0, float(volume)))

    @property
    def active(self):
        return self.running and self.output_stream is not None

    # ---------------- 远端语音 ----------------

    def push_remote(self, sender_id, pcm):
        """把一块远端 PCM 放进对应说话人的播放缓冲。"""
        if not self.running or not pcm:
            return
        with self.lock:
            buffer = self.buffers.get(sender_id)
            if buffer is None:
                if len(self.buffers) >= MAX_SENDERS:
                    return
                buffer = bytearray()
                self.buffers[sender_id] = buffer
            buffer.extend(pcm)
            # 对方声卡比本机快、或网络抖动造成积压时，丢掉最旧的一整块。
            # 这样播放始终停在“现在”，而不会越拖越久。
            while len(buffer) > MAX_BUFFER_BYTES:
                del buffer[:BLOCK_BYTES]

    def drop_remote(self, sender_id):
        """成员离开时丢掉他残留的缓冲，避免旧声音一直放在那里等播放。"""
        with self.lock:
            self.buffers.pop(sender_id, None)

    # ---------------- 音频回调 ----------------

    def _capture_callback(self, indata, frames, time_info, status):
        if status:
            print(f"音频采集状态: {status}")
        if self.muted or not self.running:
            return
        try:
            self.on_capture(np.ascontiguousarray(indata[:, 0]).tobytes())
            # ↑ indata 形状是 [frames, 1]（单声道），取第 0 列得到一维 PCM。
        except Exception as e:
            print(f"发送语音失败: {e}")

    def _playback_callback(self, outdata, frames, time_info, status):
        if status:
            print(f"音频播放状态: {status}")

        need = frames * 2
        mixed = np.zeros(frames, dtype=np.int32)

        with self.lock:
            for buffer in self.buffers.values():
                if len(buffer) < need:
                    continue
                # ↑ 这个说话人这一块没到齐就按静音处理：宁可少他半句，也不要把播放拖住。
                mixed += np.frombuffer(bytes(buffer[:need]), dtype=np.int16)
                del buffer[:need]

        audio = mixed.astype(np.float32)
        if self.volume != 1.0:
            audio *= self.volume
        np.clip(audio, -32768, 32767, out=audio)
        # ↑ 多路相加会超出 int16 范围，必须裁剪，否则爆音。
        outdata[:, 0] = audio.astype(np.int16)
