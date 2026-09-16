# -*- coding: utf-8 -*-
"""局域网多人视频会议传输层。

本模块刻意不依赖 PyQt 以外的网络库：只用 TCP、线程、队列和 OpenCV。
它面向同一局域网内的小型会议，采用一个主机（relay）转发、其余成员连接主机的星型拓扑。

为什么不用“每人连每人”的 P2P：N 个人全互连需要 N(N-1)/2 条连接，每个人还要编码/发送
N-1 份相同画面；星型拓扑中客户端只上传一份 JPEG，连接数和 CPU 消耗都可控。

协议：每个 TCP 包以 5 字节头部开始（1 字节类型 + 4 字节大端长度）。
    CONTROL = JSON 文本，用于加入、离开、成员列表、错误等控制信息。
    VIDEO   = [发送者 id 长度: uint16][id UTF-8][JPEG 数据]。
    AUDIO   = [发送者 id 长度: uint16][id UTF-8][PCM 数据]，16 kHz 单声道 16-bit。
    MOSAIC  = [布局版本: uint32][JPEG 数据]，主机合成好的一整张多宫格画面。

两种画面模式（LANConference.mode）：
    "relay"  主机把每路画面原样转给其他所有人。每人收到 N-1 路 480×270，画质好，
             但主机上行带宽随人数近似平方增长，十几人就是普通局域网的极限。
    "mosaic" 主机把所有人的画面拼成一张多宫格图，只编码一路发给所有人。上行降到
             每人一路，代价是大家看到的是同一张"缩略图墙"，单人格子会随人数变小。

两个模式下客户端都只做解码，不做合成；主机才是合成的那一方。

语音和视频一样由主机按发送者排除转发，所以谁都不会听到自己的声音。主机不做混音：
要让 N 个人各自听到一份“去掉自己”的混音，主机得编码 N 路；而客户端把收到的几路
PCM 直接相加几乎不花 CPU，代价只是带宽——在局域网里这比视频便宜得多。

TCP 保障有序可靠，但慢客户端可能堆积旧画面。因此每个连接有两个发送队列：控制与语音
进优先队列，视频单独排队且只保留最新两帧。慢连接丢的是过期画面，而不是语音。
"""

import json
import math
import queue
import socket
import struct
import threading
import time
import uuid

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


CONTROL = 1
VIDEO = 2
AUDIO = 3
MOSAIC = 4
HEADER = struct.Struct("!BI")
MAX_PACKET_BYTES = 2 * 1024 * 1024  # 防止错误或恶意客户端分配超大内存

MAX_PARTICIPANTS = 50
# ↑ 一场会议的总人数上限（含主持人）。
#   注意：relay 模式下视频链路是“主机把每一路转给其他所有人”的星型结构，主机上行带宽
#   随人数近似平方增长，十几人就已经逼近普通千兆局域网的极限。mosaic 模式把这个增长
#   降到线性，才是真正能开大会议的路径。

MAX_MOSAIC_WIDTH = 1600
# ↑ 合成画面的宽度上限。人数越多格子越小，但整张图不会无限变大——否则主机编码和
#   每个客户端的解码开销都会跟着人数一起涨。
MOSAIC_TILE_MAX = 480
# ↑ 单格最大边长。源画面上限就是 480×270，再放大没有意义，只会浪费码率。

RELAY = "relay"
MOSAIC_MODE = "mosaic"


def mosaic_layout(count, max_width=MAX_MOSAIC_WIDTH):
    """由人数算出宫格的列数、行数和单格尺寸。

    主机和每个客户端都用这一个函数，保证双方对“第 i 格是谁”的判断完全一致——
    布局不需要额外传输，只要双方的人数相同即可。
    """
    if count <= 0:
        return 1, 1, 2, 2
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / cols)
    tile_w = min(MOSAIC_TILE_MAX, max(2, max_width // cols))
    tile_w -= tile_w % 2
    tile_h = max(2, tile_w * 9 // 16)
    tile_h -= tile_h % 2
    # ↑ 源画面是 16:9，单格也按 16:9 取，拼出来不会变形；宽高都取偶数，JPEG 编码更友好。
    return cols, rows, tile_w, tile_h


def _encode_control(message):
    """把控制字典编码为一个协议包。"""
    return CONTROL, json.dumps(message, ensure_ascii=False).encode("utf-8")


def _stamp(sender_id, blob):
    """在媒体数据前写入发送者 id，供接收端识别说话人或画面来源。"""
    sender = sender_id.encode("utf-8")
    return struct.pack("!H", len(sender)) + sender + blob


def _split_sender(payload):
    """拆出 [id 长度][id][媒体数据] 中的发送者与数据；格式不对时返回 None。"""
    if len(payload) < 2:
        return None
    sender_len = struct.unpack("!H", payload[:2])[0]
    if len(payload) <= 2 + sender_len:
        return None
    sender = payload[2:2 + sender_len].decode("utf-8", errors="replace")
    return sender, payload[2 + sender_len:]


def _set_nodelay(sock):
    """关闭 Nagle 合并算法。

    语音每 20 ms 才发一小块，Nagle 会把它攒到凑满一个 MSS 再发，凭空多出几十毫秒
    延迟；视频帧虽然够大，关掉之后也不会再被前一个包拖住。
    """
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass


def _recv_exact(sock, count):
    """从 TCP 流中累计读取指定字节数；连接关闭时返回 None。"""
    chunks = bytearray()
    while len(chunks) < count:
        part = sock.recv(count - len(chunks))
        if not part:
            return None
        chunks.extend(part)
    return bytes(chunks)


class _PeerConnection:
    """一条 TCP 连接的异步发送端。

    接收由拥有它的 server/client 线程处理；发送单独排队，避免转发慢客户端时卡住
    摄像头或其他成员。

    控制与语音走 urgent 队列：语音丢一块就是一次可闻的断点，队列真的堵死时也只丢
    最旧的一块，让播放停在“现在”。视频走 video 队列且只留最新两帧——画面迟到不如
    直接跳过。
    """

    def __init__(self, sock, address, on_closed):
        self.sock = sock
        self.address = address
        self.on_closed = on_closed
        self.alive = True
        self.draining = False
        self.send_lock = threading.Lock()
        self.urgent = queue.Queue(maxsize=32)
        self.video = queue.Queue(maxsize=2)
        self.writer = threading.Thread(target=self._write_loop, daemon=True)
        self.writer.start()

    def send(self, kind, payload):
        if not self.alive:
            return
        target = self.video if kind == VIDEO else self.urgent
        item = (kind, payload)
        try:
            target.put_nowait(item)
        except queue.Full:
            # 丢掉队首那块旧数据，把当前这块塞进去，保证发出去的是最新的。
            try:
                target.get_nowait()
                target.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def _write_loop(self):
        try:
            while self.alive:
                if self.draining and self.urgent.empty() and self.video.empty():
                    break
                    # ↑ 已经没什么要发的了，这时才真正断开，见 close_after_flush()
                try:
                    item = self.urgent.get(timeout=0.5)
                except queue.Empty:
                    try:
                        self._send_one(*self.video.get_nowait())
                    except queue.Empty:
                        pass
                    continue
                self._send_one(*item)
                # 语音是持续的，urgent 可能一直非空；这里顺带送一帧视频，免得画面被饿死。
                try:
                    self._send_one(*self.video.get_nowait())
                except queue.Empty:
                    pass
        except OSError:
            pass
        finally:
            self.close()

    def close_after_flush(self):
        """等待发消息全部写出后再关连接。

        用于“会议已满”这类拒绝消息：如果发完立刻 close()，写线程往往还没把数据交给
        内核，socket 就已经被关掉，客户端只能看到连接被重置，收不到任何拒绝原因。
        """
        self.draining = True

    def _send_one(self, kind, payload):
        if len(payload) > MAX_PACKET_BYTES:
            return
        with self.send_lock:
            self.sock.sendall(HEADER.pack(kind, len(payload)) + payload)

    def close(self):
        if not self.alive:
            return
        self.alive = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        self.on_closed(self)


class _ConferenceServer:
    """会议主机。只做成员管理与视频转发，不做重编码。"""

    def __init__(self, host_name, port, max_participants, on_status, on_video, on_audio,
                 on_members, on_mosaic, mode=RELAY, fps=12, quality=55):
        self.host_id = "host-" + uuid.uuid4().hex[:12]
        self.host_name = host_name
        self.port = port
        self.max_participants = max_participants
        self.on_status = on_status
        self.on_video = on_video
        self.on_audio = on_audio
        self.on_members = on_members
        self.on_mosaic = on_mosaic
        self.mode = mode
        self.fps = fps
        self.quality = quality
        self.listener = None
        self.running = False
        self.lock = threading.RLock()
        self.peers = {}  # peer object -> {id, name}

        # 合成模式用的帧仓库：participant_id -> {"jpeg": 原始字节, "image": 解码缓存}。
        # 单独一把锁，避免主机解码几十帧时把成员管理和收发线程一起卡住。
        self.frame_lock = threading.Lock()
        self.latest = {}
        self.layout_ver = 0
        # ↑ 成员每次变化就 +1。客户端拿它判断收到的合成图是哪一版布局拼的——
        #   布局刚变时客户端手里的格子顺序和画面对不上，这一帧必须丢掉。

    def start(self):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("0.0.0.0", self.port))
        self.listener.listen(MAX_PARTICIPANTS * 2)
        # ↑ 积压队列要能容纳满员同时涌入的连接请求，否则多出来的会被系统直接拒绝。
        self.listener.settimeout(1.0)
        self.running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        if self.mode == MOSAIC_MODE:
            threading.Thread(target=self._compose_loop, daemon=True).start()
            # ↑ 合成是主机独有的活儿：把所有人的画面拼成一张再发出去。
        self._publish_members()
        label = "合成" if self.mode == MOSAIC_MODE else "转发"
        self.on_status(f"会议主机已启动，端口 {self.port}，最多 {self.max_participants} 人，"
                       f"画面模式：{label}")

    def members(self):
        with self.lock:
            result = [{"id": self.host_id, "name": self.host_name}]
            result.extend({"id": v["id"], "name": v["name"]} for v in self.peers.values())
            return result

    def _accept_loop(self):
        while self.running:
            try:
                sock, address = self.listener.accept()
                sock.settimeout(None)
                _set_nodelay(sock)
                peer = _PeerConnection(sock, address, self._peer_closed)
                threading.Thread(target=self._read_peer, args=(peer,), daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _read_peer(self, peer):
        try:
            while self.running and peer.alive:
                header = _recv_exact(peer.sock, HEADER.size)
                if header is None:
                    break
                kind, length = HEADER.unpack(header)
                if length > MAX_PACKET_BYTES:
                    break
                payload = _recv_exact(peer.sock, length)
                if payload is None:
                    break
                if kind == CONTROL:
                    self._handle_control(peer, payload)
                elif kind == VIDEO:
                    self._handle_video(peer, payload)
                elif kind == AUDIO:
                    self._handle_audio(peer, payload)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        finally:
            peer.close()

    def _handle_control(self, peer, payload):
        message = json.loads(payload.decode("utf-8"))
        if message.get("action") != "hello" or peer in self.peers:
            return
        name = str(message.get("name", "访客")).strip()[:32] or "访客"
        with self.lock:
            if len(self.peers) + 1 >= self.max_participants:
                peer.send(*_encode_control({
                    "action": "error",
                    "message": f"会议已满（上限 {self.max_participants} 人）"
                }))
                peer.close_after_flush()
                # ↑ 必须等拒绝消息发出去再断，否则对方只会看到连接被重置。
                return
            self.peers[peer] = {"id": "peer-" + uuid.uuid4().hex[:12], "name": name}
            peer.send(*_encode_control({
                "action": "welcome",
                "id": self.peers[peer]["id"],
                "ver": self.layout_ver,
                # ↑ 带上当前布局版本，新成员在收到第一条 members 之前也能对上传来的合成图。
            }))
        self._publish_members()
        self.on_status(f"{name} 已加入会议")

    def _handle_video(self, peer, jpeg):
        with self.lock:
            info = self.peers.get(peer)
        if info is None or not jpeg:
            return
        if self.mode == MOSAIC_MODE:
            self._store_frame(info["id"], jpeg)
            return
        # 发送者无需接收自己的回环帧；主机本地 UI 则通过回调直接显示。
        self._broadcast(VIDEO, _stamp(info["id"], jpeg), exclude=peer)
        self.on_video(info["id"], info["name"], jpeg)

    def _handle_audio(self, peer, pcm):
        with self.lock:
            info = self.peers.get(peer)
        if info is None or not pcm:
            return
        # 与视频同理：不回发给发送者本人，否则他会听到自己的声音延迟回来。
        self._broadcast(AUDIO, _stamp(info["id"], pcm), exclude=peer)
        self.on_audio(info["id"], info["name"], pcm)

    def publish_host_video(self, jpeg):
        """主机自己的画面：转发模式下直接转发，合成模式下只存进帧仓库。"""
        if self.mode == MOSAIC_MODE:
            self._store_frame(self.host_id, jpeg)
            return
        self._broadcast(VIDEO, _stamp(self.host_id, jpeg))

    def publish_host_audio(self, pcm):
        """主机自己的语音直接转发给所有客户端。"""
        self._broadcast(AUDIO, _stamp(self.host_id, pcm))

    # ---------------- 画面合成（mosaic 模式）----------------

    def _store_frame(self, participant_id, jpeg):
        """收下一路画面。只存字节，解码推迟到合成时按需做。"""
        with self.frame_lock:
            entry = self.latest.get(participant_id)
            if entry is None:
                if len(self.latest) >= MAX_PARTICIPANTS:
                    return
                entry = {"jpeg": None, "image": None, "image_src": None}
                self.latest[participant_id] = entry
            entry["jpeg"] = jpeg
            # ↑ 不动 image/image_src：合成线程会用 image_src 判断缓存是否还对应当前这帧。

    def _compose_loop(self):
        interval = 1.0 / max(1, self.fps)
        while self.running:
            time.sleep(interval)
            if not self.running:
                break
            try:
                self._compose_once()
            except Exception as e:
                print(f"合成画面失败: {e}")

    def _compose_once(self):
        """把每个人的最新画面拼成一张多宫格图，编码一次发给所有人。"""
        with self.lock:
            members = self.members()
            ver = self.layout_ver
        if not members:
            return

        cols, rows, tile_w, tile_h = mosaic_layout(len(members))

        # 先在锁里取快照，解码放到锁外面做：几十帧解码要几十毫秒，攥着锁会把
        # 所有收帧线程一起堵住。
        with self.frame_lock:
            snapshot = []
            for member in members:
                entry = self.latest.get(member["id"])
                if entry is not None and entry["jpeg"]:
                    snapshot.append((member["id"], entry["jpeg"],
                                     entry["image"], entry["image_src"]))

        decoded, fresh = {}, {}
        for participant_id, jpeg, image, image_src in snapshot:
            if image is not None and image_src is jpeg:
                decoded[participant_id] = image
                continue
            # ↑ 缓存对应当前这帧才复用，否则重新解码。
            try:
                new_image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8),
                                         cv2.IMREAD_COLOR)
            except cv2.error:
                new_image = None
            if new_image is not None:
                decoded[participant_id] = new_image
                fresh[participant_id] = (new_image, jpeg)

        with self.frame_lock:
            for participant_id, (image, jpeg) in fresh.items():
                entry = self.latest.get(participant_id)
                if entry is not None and entry["jpeg"] is jpeg:
                    # 只有这帧还没被新帧顶掉才写回缓存，避免把旧解码结果盖在新帧上。
                    entry["image"] = image
                    entry["image_src"] = jpeg

        canvas = np.full((rows * tile_h, cols * tile_w, 3), 32, dtype=np.uint8)
        # ↑ 还没出画面（或人刚走）的格子留深灰底色，比黑块好看。
        for index, member in enumerate(members):
            image = decoded.get(member["id"])
            if image is None:
                continue
            row, col = divmod(index, cols)
            y, x = row * tile_h, col * tile_w
            canvas[y:y + tile_h, x:x + tile_w] = cv2.resize(
                image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

        try:
            ok, encoded = cv2.imencode(".jpg", canvas,
                                       [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        except cv2.error:
            ok = False
        if not ok:
            return

        self._broadcast(MOSAIC, struct.pack("!I", ver) + encoded.tobytes())
        self.on_mosaic(ver, canvas)

    def _broadcast(self, kind, payload, exclude=None):
        with self.lock:
            targets = list(self.peers)
        for peer in targets:
            if peer is not exclude:
                peer.send(kind, payload)

    def _publish_members(self):
        with self.lock:
            self.layout_ver += 1
            ver = self.layout_ver
            members = self.members()
        # ↑ 成员一变，布局版本就 +1 一起发下去；客户端据此判断手里的格子顺序还对不对。
        self._broadcast(*_encode_control(
            {"action": "members", "members": members, "ver": ver}))
        self.on_members(members, ver)

    def _peer_closed(self, peer):
        with self.lock:
            info = self.peers.pop(peer, None)
        if info:
            with self.frame_lock:
                self.latest.pop(info["id"], None)
                # ↑ 人走了就把他的画面丢掉，否则他那一格会一直停在最后一帧。
            self.on_status(f"{info['name']} 已离开会议")
            self._publish_members()

    def stop(self):
        self.running = False
        if self.listener:
            try:
                self.listener.close()
            except OSError:
                pass
        with self.lock:
            peers = list(self.peers)
            self.peers.clear()
        with self.frame_lock:
            self.latest.clear()
        for peer in peers:
            peer.close()


class LANConference(QObject):
    """供 PyQt 主窗口使用的会议控制器。

    调用 host() 或 join() 后，将每帧合成画面传给 publish_frame()。接收到的远端画面
    通过 remote_frame 信号回到 GUI 线程，调用者不必处理网络线程同步问题。
    """

    remote_frame = pyqtSignal(str, str, np.ndarray)  # participant_id, display_name, BGR
    remote_audio = pyqtSignal(str, str, object)      # participant_id, display_name, PCM bytes
    mosaic_frame = pyqtSignal(int, np.ndarray)       # layout_ver, 主机合成好的 BGR 大图
    members_changed = pyqtSignal(object)              # list[dict]
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.server = None
        self.client = None
        self.client_peer = None
        self.local_id = None
        self.local_name = "我"
        self.member_names = {}
        self.connected = False
        self.last_sent_at = 0.0
        self.fps = 12
        self.quality = 55
        self.frame_size = (480, 270)  # 宽、高；缩小后再压缩，每路的带宽与解码成本都可控
        self.mode = RELAY
        # ↑ "relay" 主机逐路转发（画质好、带宽随人数平方增长）；
        #   "mosaic" 主机合成一张多宫格图（带宽线性、大家看到同一张缩略图墙）。
        #   由界面选择，只在“创建主机”时生效。
        self.layout_ver = 0

    @property
    def is_host(self):
        return self.server is not None

    def host(self, name, port, max_participants=MAX_PARTICIPANTS):
        self.leave()
        self.local_name = name.strip()[:32] or "主持人"
        # 至少两人（主机 + 一名成员）才有会议的意义；上限由界面上的“人数上限”决定。
        capacity = max(2, int(max_participants))
        try:
            self.server = _ConferenceServer(
                self.local_name, int(port), capacity,
                self.status.emit, self._receive_jpeg, self._receive_audio,
                self._members_received, self._receive_mosaic,
                mode=self.mode, fps=self.fps, quality=self.quality
            )
            self.local_id = self.server.host_id
            # ↑ 必须在 start() 之前设好：start() 会立刻发布成员列表并回调
            #   members_changed，那时若 local_id 还是 None，主机就会把自己也当成
            #   远端成员，在自己的成员网格里画出一个多余的格子。
            self.server.start()
            self.connected = True
        except OSError as exc:
            self.server = None
            self.local_id = None
            self.error.emit(f"无法启动会议主机：{exc}")

    def join(self, host, port, name):
        self.leave()
        self.local_name = name.strip()[:32] or "访客"
        try:
            sock = socket.create_connection((host.strip(), int(port)), timeout=5)
            sock.settimeout(None)
            _set_nodelay(sock)
            self.client_peer = _PeerConnection(sock, (host, int(port)), self._client_closed)
            self.connected = True
            self.client_peer.send(*_encode_control({"action": "hello", "name": self.local_name}))
            threading.Thread(target=self._read_client, daemon=True).start()
            self.status.emit(f"已连接到会议主机 {host}:{port}")
        except OSError as exc:
            self.client_peer = None
            self.connected = False
            self.error.emit(f"无法连接会议主机：{exc}")

    def publish_frame(self, frame_bgr):
        """压缩并上传本地画面。调用频率不限，内部会按 fps 节流。"""
        if not self.connected or frame_bgr is None or frame_bgr.size == 0:
            return
        now = time.monotonic()
        if now - self.last_sent_at < 1.0 / max(1, self.fps):
            return
        self.last_sent_at = now
        try:
            # 先缩到传输尺寸，显著降低多人会议所需带宽和接收端解码成本。
            resized = cv2.resize(frame_bgr, self.frame_size, interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ok:
                return
            jpeg = encoded.tobytes()
            if self.server:
                self.server.publish_host_video(jpeg)
            elif self.client_peer:
                self.client_peer.send(VIDEO, jpeg)
        except (cv2.error, OSError):
            pass

    def publish_audio(self, pcm):
        """上传一块本地采集的语音。

        由音频采集回调直接调用，因此这里不做任何节流：采集回调本身就是按 20 ms 一块
        触发的，再节流只会把语音切碎。
        """
        if not self.connected or not pcm:
            return
        try:
            if self.server:
                self.server.publish_host_audio(pcm)
            elif self.client_peer:
                self.client_peer.send(AUDIO, pcm)
        except OSError:
            pass

    def _read_client(self):
        peer = self.client_peer
        if peer is None:
            return
        try:
            while self.connected and peer.alive:
                header = _recv_exact(peer.sock, HEADER.size)
                if header is None:
                    break
                kind, length = HEADER.unpack(header)
                if length > MAX_PACKET_BYTES:
                    break
                payload = _recv_exact(peer.sock, length)
                if payload is None:
                    break
                if kind == CONTROL:
                    self._client_control(json.loads(payload.decode("utf-8")))
                elif kind == VIDEO:
                    self._client_media(payload, self._client_video)
                elif kind == AUDIO:
                    self._client_media(payload, self._client_audio)
                elif kind == MOSAIC:
                    self._client_mosaic(payload)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        finally:
            peer.close()

    def _client_control(self, message):
        action = message.get("action")
        if action == "welcome":
            self.local_id = message.get("id")
            self.layout_ver = message.get("ver", 0)
        elif action == "members":
            self._members_received(message.get("members", []), message.get("ver", 0))
        elif action == "error":
            self.error.emit(str(message.get("message", "会议连接失败")))

    def _client_mosaic(self, payload):
        """收到主机合成的一整张画面。"""
        if len(payload) <= 4:
            return
        ver = struct.unpack("!I", payload[:4])[0]
        if ver != self.layout_ver:
            return
            # ↑ 布局刚变过：这一帧是按旧顺序拼的，格子会对错人，直接丢掉。
        try:
            image = cv2.imdecode(np.frombuffer(payload[4:], dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
        except cv2.error:
            return
        if image is not None:
            self.mosaic_frame.emit(ver, image)

    def _client_media(self, payload, handle):
        """拆出语音/视频的发送者后交给对应处理函数。"""
        split = _split_sender(payload)
        if split is None:
            return
        sender, blob = split
        handle(sender, self.member_names.get(sender, "参会者"), blob)

    def _client_video(self, sender, name, jpeg):
        self._receive_jpeg(sender, name, jpeg)

    def _client_audio(self, sender, name, pcm):
        self._receive_audio(sender, name, pcm)

    def _receive_jpeg(self, participant_id, name, jpeg):
        try:
            image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                self.remote_frame.emit(participant_id, name, image)
        except cv2.error:
            pass

    def _receive_audio(self, participant_id, name, pcm):
        self.remote_audio.emit(participant_id, name, pcm)

    def _receive_mosaic(self, ver, image):
        """主机自己合成出来的那张图，直接送到本地界面。"""
        self.mosaic_frame.emit(ver, image)

    def _members_received(self, members, ver=0):
        self.layout_ver = ver
        self.member_names = {str(m["id"]): str(m["name"]) for m in members if "id" in m and "name" in m}
        self.members_changed.emit(members)

    def _client_closed(self, peer):
        if peer is self.client_peer and self.connected:
            self.connected = False
            self.status.emit("已与会议主机断开连接")

    def leave(self):
        self.connected = False

        if self.server:
            self.server.stop()
        if self.client_peer:
            self.client_peer.close()
        self.server = None
        self.client_peer = None
        self.local_id = None
        self.layout_ver = 0
        self.member_names = {}
        self.members_changed.emit([])

