# -*- coding: utf-8 -*-
# ↑ 声明文件编码为 UTF-8，确保 Python 解释器能正确处理中文字符。
#   在 Python 3 中默认已经是 UTF-8，但写上更保险，特别是 Windows 环境。

"""
================================================================================
【文件总说明】
--------------------------------------------------------------------------------
文件名    : virtual_background_u2net.py
功能      : 基于 U²-Net 的实时虚拟背景程序，带 PyQt5 GUI 和声音录制
作者      : 王宏
运行环境  : Windows 10/11 + Python 3.10 + PyTorch 2.x + CUDA（可选）
依赖      : PyQt5、OpenCV、Pillow、numpy、torch、sounddevice（可选）、ffmpeg

【程序运行时的数据流】
  摄像头帧(BGR) → 水平镜像(可选) → 预处理(RGB/缩放到320/PIL归一化)
  → 送入 U²-Net(CPU/GPU) → 得到前景 logits
  → sigmoid → 概率图 → 用 PIL 上采样回摄像头尺寸
  → 阈值二值化 → 形态学开运算去噪 → 高斯模糊羽化
  → 得到 alpha 掩码 → 与背景做 Alpha 合成 → 显示/保存/录制

【主要线程/进程】
  1) 主线程(Qt 事件循环)：处理 GUI 事件、刷新预览、更新状态栏
  2) VideoThread(QThread)：独立线程，循环读摄像头帧、跑模型、发信号
  3) AudioRecorder(sounddevice 回调线程)：本地录制时采集麦克风数据
  4) ffmpeg 子进程：录制结束时合并视频与音频
  5) 会议网络线程 + AudioLink 音频回调线程：收发局域网会议的画面与语音

【关键设计原则】
  - GUI 不能卡：所有耗时操作放到 QThread 里
  - 宁可丢帧不卡顿：每 2 帧推理一次，未推理时复用上一次掩码
  - 安全第一：任何路径、文件、设备都要先校验再用
  - 异常不崩溃：所有可能出错的地方都用 try/except 包住
================================================================================
"""

# ==============================================================================
# 【模块 1：标准库导入】
# 标准库是 Python 自带的，不需要 pip 安装。按字母顺序排列是 Python 社区惯例。
# ==============================================================================

import sys
# ↑ sys 模块：提供与 Python 解释器交互的接口。
#   本程序主要用：
#     sys.argv  → 获取命令行参数（本程序没有用，但保留习惯）
#     sys.exit  → 退出程序并返回状态码

import os
# ↑ os 模块：操作系统接口。
#   本程序主要用：
#     os.path.isfile / os.path.exists → 判断文件是否存在
#     os.makedirs                     → 创建目录（含中间目录）
#     os.remove                       → 删除文件
#     os.rename                       → 重命名文件
#     os.name                         → 判断是 Windows('nt') 还是 Linux('posix')
#     os.getcwd                       → 获取当前工作目录

import time
# ↑ time 模块：时间相关。
#   本程序主要用：
#     time.time()  → 返回当前时间戳（浮点秒），用于 FPS 和推理计时
#     time.sleep() → 让当前线程暂停一段时间，用于空帧时避免死循环

import wave
# ↑ wave 模块：读写 WAV 音频文件。
#   本程序主要用：
#     wave.open(path, 'wb')    → 以写模式打开 WAV
#     wf.setnchannels(n)       → 设置声道数（1=单声道，2=立体声）
#     wf.setsampwidth(n)       → 设置采样位宽（2 = 16 bit）
#     wf.setframerate(n)       → 设置采样率（如 44100 Hz）
#     wf.writeframes(bytes)    → 写入 PCM 数据

import shutil
# ↑ shutil 模块：高级文件操作。
#   本程序主要用：
#     shutil.which("ffmpeg") → 在系统 PATH 中查找可执行文件路径

import socket
# ↑ 用于推导本机的局域网 IPv4 地址，方便其他参会者填写主机 IP。

import threading
# ↑ threading 模块：线程相关。
#   本程序主要用：
#     threading.Lock() → 创建互斥锁，保护 AudioRecorder.frames 列表

import subprocess
# ↑ subprocess 模块：启动外部进程。
#   本程序主要用：
#     subprocess.run(cmd, ...) → 执行 ffmpeg 命令

import traceback
# ↑ traceback 模块：异常堆栈打印。
#   本程序主要用：
#     traceback.print_exc() → 打印完整异常堆栈到控制台

from datetime import datetime
# ↑ 从 datetime 模块导入 datetime 类（而不是整个模块）。
#   本程序主要用：
#     datetime.now().strftime("%Y%m%d_%H%M%S") → 生成时间戳文件名

# 局域网会议传输层独立放在 lan_conference.py：网络协议的修改不会影响 U²-Net 推理。
from lan_conference import (
    LANConference, MAX_PARTICIPANTS, MOSAIC_MODE, RELAY, mosaic_layout
)

# ==============================================================================
# 【模块 2：第三方库导入】
# 这些需要 pip 安装：pip install opencv-python numpy torch torchvision pillow
# ==============================================================================

import cv2
# ↑ OpenCV：计算机视觉库。
#   本程序主要用：
#     cv2.VideoCapture         → 打开摄像头
#     cv2.cvtColor             → 颜色空间转换（BGR↔RGB）
#     cv2.flip                 → 水平镜像
#     cv2.morphologyEx         → 形态学操作（开运算去噪）
#     cv2.GaussianBlur         → 高斯模糊
#     cv2.VideoWriter          → 写视频文件
#     cv2.VideoWriter_fourcc   → 生成 FourCC 编码标识
#     cv2.imencode             → 把图像编码成 JPG/PNG 内存数据

import numpy as np
# ↑ NumPy：数值计算库。
#   本程序主要用：
#     np.array / np.asarray    → 创建数组
#     np.full                  → 创建填充相同值的数组（纯色背景）
#     np.concatenate           → 拼接音频分块
#     np.clip                  → 数值裁剪
#     np.isnan                 → 检查 NaN
#     np.float32 / np.uint8    → 指定数据类型

import torch
# ↑ PyTorch：深度学习框架。
#   本程序主要用：
#     torch.load               → 加载权重
#     torch.device             → 指定 CPU/GPU
#     torch.no_grad()          → 关闭梯度计算
#     torch.cuda.amp.autocast  → FP16 自动混合精度
#     torch.cuda.synchronize   → 等待 GPU 完成计算（用于准确计时）
#     torch.sigmoid            → sigmoid 激活
#     torch.from_numpy         → numpy 转 tensor

import torch.nn as nn
# ↑ torch.nn：神经网络模块。本程序用 nn.Module、nn.Conv2d、nn.BatchNorm2d、
#   nn.ReLU、nn.MaxPool2d 等来构建 U²-Net。

import torch.nn.functional as F
# ↑ torch.nn.functional：函数式接口。本程序用 F.interpolate 做双线性上采样。

from PIL import Image
# ↑ Pillow：图像处理库。
#   本程序主要用：
#     Image.open               → 打开图片
#     Image.fromarray          → numpy 数组转 PIL 图像
#     Image.BILINEAR           → 双线性插值
#     img.resize(...)          → 缩放（避开 OpenCV 5.0.0 HAL bug）

# ==============================================================================
# 【模块 3：可选依赖 sounddevice】
# sounddevice 不是必须的：如果用户不录音，程序也能运行。
# 因此用 try/except 包住导入，并用一个布尔变量标记是否可用。
# ==============================================================================

try:
    import sounddevice as sd
    # ↑ sounddevice 是基于 PortAudio 的音频库，支持跨平台采集/播放。
    HAS_SOUNDDEVICE = True
    # ↑ 标记变量：后续代码通过它判断能否录音
except Exception:
    # ↑ 这里捕获所有异常（ImportError 或 PortAudio 初始化失败等）
    HAS_SOUNDDEVICE = False

# ==============================================================================
# 【模块 4：PyQt5 导入】
# ==============================================================================

from PyQt5.QtCore import Qt, QThread, pyqtSignal
# ↑ Qt              : 枚举常量，如 Qt.Horizontal、Qt.AlignCenter、Qt.KeepAspectRatio
#   QThread         : Qt 线程基类，VideoThread 继承它
#   pyqtSignal      : 定义信号的工厂函数

from PyQt5.QtGui import QImage, QPixmap, QColor
# ↑ QImage          : Qt 图像对象（CPU 侧）
#   QPixmap         : Qt 像素图（显示优化）
#   QColor          : 颜色对象

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QComboBox, QSlider, QCheckBox, QRadioButton, QButtonGroup,
    QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QFormLayout,
    QGroupBox, QColorDialog, QStatusBar, QGridLayout,
    QScrollArea, QFrame, QSpinBox
)
# ↑ QApplication    : Qt 应用对象，每个 Qt 程序必须有一个
#   QMainWindow     : 主窗口基类
#   QWidget         : 通用控件/容器
#   QLabel          : 文字/图片标签
#   QPushButton     : 按钮
#   QLineEdit       : 单行文本输入
#   QComboBox       : 下拉框
#   QSlider         : 滑块
#   QCheckBox       : 复选框
#   QRadioButton    : 单选按钮
#   QButtonGroup    : 单选按钮分组（互斥）
#   QFileDialog     : 文件选择对话框
#   QMessageBox     : 消息弹窗（警告/错误/信息）
#   QHBoxLayout     : 水平布局
#   QVBoxLayout     : 垂直布局
#   QFormLayout     : 表单布局（左标签右控件）
#   QGroupBox       : 分组框（带标题）
#   QColorDialog    : 颜色选择对话框
#   QStatusBar      : 状态栏
#   QScrollArea     : 滚动区域，左侧面板控件太多时保证不被窗口高度截断
#   QFrame          : 控件基类，这里只用它的 NoFrame 枚举去掉滚动区边框
#   QSpinBox        : 整数输入框，用于会议人数上限

# 会议语音的本地收发层：麦克风采集 + 扬声器混音播放。
from audio_link import AudioLink, list_devices

REMOTE_GRID_COLUMNS = 6
# ↑ 远端成员网格的列数。人数上限放宽后行数会明显变多，因此网格外还套了一层滚动区。

# ==============================================================================
# 【模块 5：U²-Net 模型定义】
# ------------------------------------------------------------------------------
# U²-Net 论文：https://arxiv.org/pdf/2005.09007.pdf
# 核心思想：把 U-Net 的每个阶段替换成一个 RSU（Residual U-block），
# 从而在每一级都能提取多尺度特征，同时保持轻量。
# ==============================================================================

class REBNCONV(nn.Module):
    """
    REBNCONV = Residual Encoder Block with CONV
    这是 U²-Net 中最基础的构建块，依次执行：
        1. Conv2d(3×3, padding=dirate, dilation=dirate)  ← 空洞卷积
        2. BatchNorm2d                                    ← 批归一化
        3. ReLU                                           ← 激活函数

    参数说明：
        in_ch   : 输入通道数
        out_ch  : 输出通道数
        dirate  : 膨胀率（dilation rate）。
                  当 dirate=1 时是普通 3×3 卷积；
                  当 dirate>1 时空洞卷积，能在不增加参数的情况下扩大感受野。
                  这在 U²-Net 的最深几层（RSU4F）用得多。
    """
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        # ↑ 构造方法：定义网络层的结构（不涉及前向计算）
        super(REBNCONV, self).__init__()
        # ↑ 调用父类 nn.Module 的构造方法。
        #   PyTorch 规定：所有自定义 Module 都必须先调用 super().__init__()，
        #   否则后续 self.xxx = nn.Conv2d(...) 的注册机制不会生效。

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                 padding=1 * dirate, dilation=1 * dirate)
        # ↑ 3×3 卷积层。
        #   padding = dirate 是空洞卷积的经典配置：
        #   当 dilation=d 时，卷积核的有效感受野为 (2d+1)×(2d+1)，
        #   为了让输出尺寸与输入一致，padding 需设为 d。
        #   kernel_size 参数没显式写 3，是因为第 3 个位置参数就是 3。

        self.bn_s1 = nn.BatchNorm2d(out_ch)
        # ↑ 批归一化层：
        #   在训练时用 batch 统计量，推理时用滑动平均统计量。
        #   model.eval() 会切换到滑动平均，这对单人推理很重要。

        self.relu_s1 = nn.ReLU(inplace=True)
        # ↑ ReLU 激活函数。
        #   inplace=True 表示直接修改输入张量的内存，节省显存。
        #   注意：inplace 后原始输入就不能再用了，所以本类内部不保留 x 引用。

    def forward(self, x):
        # ↑ 前向传播：定义数据如何流经这些层。
        #   参数 x：形状 [N, in_ch, H, W] 的张量。
        #   返回：形状 [N, out_ch, H, W] 的张量。

        return self.relu_s1(self.bn_s1(self.conv_s1(x)))
        # ↑ 从内到外依次执行：卷积 → BN → ReLU。
        #   等价于：
        #       y = self.conv_s1(x)
        #       y = self.bn_s1(y)
        #       y = self.relu_s1(y)
        #       return y


# ------------------------------------------------------------------------------
# 【RSU7：最深的一级编码块】
# ------------------------------------------------------------------------------
class RSU7(nn.Module):
    """
    RSU-7：编码器最深的 U-block，包含 7 层卷积（含输入和输出）。
    结构：
        输入 → rebnconvin → [编码器 6 次下采样] → rebnconv7(最深处)
             → [解码器 6 次上采样+跳跃连接] → 残差相加

    参数：
        in_ch   : 输入通道
        mid_ch  : 中间层通道（每一层的“主干”通道数）
        out_ch  : 输出通道
    """
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        # ---- 输入通道映射 ----
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        # ↑ 把输入通道数变换成 out_ch，方便后续拼接使用

        # ---- 编码路径（下采样 5 次）----
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        # ↑ 第 1 级特征，尺寸与输入相同
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # ↑ 2×2 最大池化，stride=2 使尺寸减半。
        #   ceil_mode=True 保证奇数尺寸时向上取整，这是 U²-Net 官方配置。

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # ---- 最深层（不再池化，改用空洞卷积）----
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        # ↑ dirate=2 的空洞卷积，在不增加参数的前提下扩大感受野，
        #   让最深层能"看到"更大的上下文。

        # ---- 解码路径（上采样 5 次 + 跳跃拼接）----
        # 每次拼接后通道数变成 mid_ch*2，所以卷积输入通道写 mid_ch*2
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        # ↑ 最后一层把通道数从 mid_ch 变回 out_ch

    def forward(self, x):
        hx = x
        # ↑ 保存原始输入引用，最后做残差相加用。
        #   注意此处没有 clone，因为后面没有 inplace 修改 hx。

        hxin = self.rebnconvin(hx)
        # ↑ 输入通道映射

        # ===== 编码器：逐级降分辨率 =====
        hx1 = self.rebnconv1(hxin)        # 尺度 1
        hx = self.pool1(hx1)              # 尺度 1/2
        hx2 = self.rebnconv2(hx)          # 尺度 1/2
        hx = self.pool2(hx2)              # 尺度 1/4
        hx3 = self.rebnconv3(hx)          # 尺度 1/4
        hx = self.pool3(hx3)              # 尺度 1/8
        hx4 = self.rebnconv4(hx)          # 尺度 1/8
        hx = self.pool4(hx4)              # 尺度 1/16
        hx5 = self.rebnconv5(hx)          # 尺度 1/16
        hx = self.pool5(hx5)              # 尺度 1/32
        hx6 = self.rebnconv6(hx)          # 尺度 1/32
        hx7 = self.rebnconv7(hx6)         # 尺度 1/32，最深层

        # ===== 解码器：逐级恢复分辨率 =====
        # 每一步都是：与对应编码层拼接 → 卷积 → 上采样（除最后一步）
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        # ↑ torch.cat(..., dim=1) 沿通道维拼接，形状从 [N,C,H,W]+[N,C,H,W] → [N,2C,H,W]

        hx6dup = F.interpolate(hx6d, size=hx5.shape[2:],
                               mode='bilinear', align_corners=True)
        # ↑ 双线性插值上采样到 hx5 的空间尺寸。
        #   size=hx5.shape[2:] 即 (H, W)。
        #   align_corners=True 是 PyTorch 官方推荐，减少错位。
        #   为什么不用转置卷积：interpolate 更简单稳定，且 U²-Net 原论文用的是它。

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:],
                               mode='bilinear', align_corners=True)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:],
                               mode='bilinear', align_corners=True)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:],
                               mode='bilinear', align_corners=True)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:],
                               mode='bilinear', align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin
        # ↑ 残差连接：把最终结果与最开始的输入相加。
        #   这样做的好处：
        #   1) 梯度能直接传回去，缓解梯度消失
        #   2) 网络只需要学习残差（差异），训练更稳定


# ------------------------------------------------------------------------------
# 【RSU6：比 RSU7 少一层】
# 结构、注释与 RSU7 相同，只是下采样次数少一次（5 次池化），编码路径 6 层。
# ------------------------------------------------------------------------------
class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)   # 最深层
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        # 编码：5 次下采样
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)
        # 解码：4 次上采样 + 拼接
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=True)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=True)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


# ------------------------------------------------------------------------------
# 【RSU5：中级 U-block】
# 池化 3 次，编码 4 层。
# ------------------------------------------------------------------------------
class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)   # 最深层用空洞卷积
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=True)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


# ------------------------------------------------------------------------------
# 【RSU4：较浅的 U-block】
# 池化 2 次，编码 3 层。
# ------------------------------------------------------------------------------
class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


# ------------------------------------------------------------------------------
# 【RSU4F：全空洞卷积版 U-block】
# 不做任何池化，只靠不同膨胀率的空洞卷积扩大感受野，
# 保持特征图分辨率不变。用于编码器的最深两层（stage5、stage6）。
# ------------------------------------------------------------------------------
class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)   # 膨胀 2
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)   # 膨胀 4
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)   # 膨胀 8
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        # 因为每层分辨率相同，直接拼接无需上采样
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        return hx1d + hxin


# ------------------------------------------------------------------------------
# 【U²-Net 完整网络】
# ------------------------------------------------------------------------------
class U2NET(nn.Module):
    """
    完整 U²-Net。
    输入  : [N, 3, H, W]（建议 H=W=320）
    输出  : 7 个张量，均为 [N, 1, H, W]
            d0 是最终融合结果，程序只使用 d0；
            d1~d6 是中间侧输出，训练时可做深监督（deep supervision）。
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()

        # ==================== 编码器 ====================
        self.stage1 = RSU7(in_ch, 32, 64)
        # ↑ 输入 3 通道 → 输出 64 通道，全分辨率
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        # ↑ 输入 64 → 输出 128，1/2 分辨率
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        # ↑ 输入 128 → 输出 256，1/4 分辨率
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        # ↑ 输入 256 → 输出 512，1/8 分辨率
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        # ↑ 1/16 分辨率
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)
        # ↑ 1/32 分辨率，最深层

        # ==================== 解码器 ====================
        # 每级输入通道 = 上一层输出(512) + 上采样(512) = 1024
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        # ==================== 侧输出 ====================
        # 每个侧输出都把特征图压成 1 通道（前景 logits）
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        # ==================== 融合卷积 ====================
        # 把 6 个侧输出沿通道拼接（6*out_ch 通道）→ 1×1 卷积融合 → out_ch
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        # ============ 编码器路径 ============
        hx = x
        hx1 = self.stage1(hx)                    # [N, 64, H,   W]
        hx = self.pool12(hx1)                    # [N, 64, H/2, W/2]
        hx2 = self.stage2(hx)                    # [N, 128,H/2, W/2]
        hx = self.pool23(hx2)                    # [N, 128,H/4, W/4]
        hx3 = self.stage3(hx)                    # [N, 256,H/4, W/4]
        hx = self.pool34(hx3)                    # [N, 256,H/8, W/8]
        hx4 = self.stage4(hx)                    # [N, 512,H/8, W/8]
        hx = self.pool45(hx4)                    # [N, 512,H/16,W/16]
        hx5 = self.stage5(hx)                    # [N, 512,H/16,W/16]
        hx = self.pool56(hx5)                    # [N, 512,H/32,W/32]
        hx6 = self.stage6(hx)                    # [N, 512,H/32,W/32]

        # ============ 解码器路径（上采样 + 跳跃拼接）============
        hx6up = F.interpolate(hx6, size=hx5.shape[2:],
                              mode='bilinear', align_corners=True)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:],
                               mode='bilinear', align_corners=True)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:],
                               mode='bilinear', align_corners=True)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:],
                               mode='bilinear', align_corners=True)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:],
                               mode='bilinear', align_corners=True)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # ============ 侧输出（全部上采样回输入尺寸）============
        d1 = self.side1(hx1d)
        # ↑ d1 已经是输入尺寸，不用上采样

        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, size=x.shape[2:],
                           mode='bilinear', align_corners=True)
        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, size=x.shape[2:],
                           mode='bilinear', align_corners=True)
        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, size=x.shape[2:],
                           mode='bilinear', align_corners=True)
        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, size=x.shape[2:],
                           mode='bilinear', align_corners=True)
        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, size=x.shape[2:],
                           mode='bilinear', align_corners=True)

        # ============ 融合：拼接 → 1×1 卷积 ============
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        # ↑ 沿通道维拼接，形状 [N, 6, H, W]，然后 1×1 卷积 → [N, 1, H, W]

        return d0, d1, d2, d3, d4, d5, d6
        # ↑ 返回 7 个张量，实际使用只取 d0


# ==============================================================================
# 【模块 6：工具函数】
# ==============================================================================

def load_u2net_model(weights_path, device):
    """
    加载 U²-Net 权重，兼容多种保存格式。

    参数:
        weights_path : .pth 权重文件路径
        device       : torch.device 对象（cuda 或 cpu）

    返回:
        model : 已加载权重、处于 eval 模式、位于指定设备上的 U²-Net
    """
    model = U2NET(in_ch=3, out_ch=1)
    # ↑ 实例化网络结构（参数是随机初始化的）

    ckpt = torch.load(weights_path, map_location=device)
    # ↑ 加载权重文件。
    #   map_location=device 告诉 PyTorch 直接把权重映射到指定设备上，
    #   避免先加载到 CPU 再搬运，节省时间和显存。

    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
        # ↑ 兼容 {'model': state_dict, 'epoch':..., 'val_iou':...} 的保存格式
    else:
        sd = ckpt
        # ↑ 普通 state_dict 格式

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v
            # ↑ 去掉 "module." 前缀（DataParallel 保存时会加）
        else:
            new_sd[k] = v
    # ↑ 构造一个干净的 state_dict，键名与 U2NET 类中的命名一致

    model.load_state_dict(new_sd, strict=True)
    # ↑ strict=True 表示键名必须完全匹配，否则报错。
    #   这是好事：如果模型结构不匹配，立即暴露，避免"看起来能跑但结果全错"。

    model.to(device)
    # ↑ 把模型参数搬到指定设备

    model.eval()
    # ↑ 切到推理模式：
    #   1) BatchNorm 使用滑动平均统计量而非 batch 统计量
    #   2) Dropout 关闭
    #   对单人推理至关重要，否则 BN 会因 batch=1 而不稳定。

    return model


def preprocess_frame(frame_bgr, input_size):
    """
    预处理摄像头帧，转成模型输入张量。

    参数:
        frame_bgr  : OpenCV 的 BGR 图像，形状 [H, W, 3]，dtype=uint8
        input_size : 模型输入边长（如 320）

    返回:
        tensor : [1, 3, input_size, input_size]，dtype=float32
    """
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # ↑ OpenCV 默认 BGR，模型期望 RGB，先转换

    img_pil = Image.fromarray(img)
    # ↑ numpy 数组 → PIL Image，以便使用 PIL 的 resize

    img_pil = img_pil.resize((input_size, input_size), Image.BILINEAR)
    # ↑ 用 PIL 缩放而不是 cv2.resize，避开 OpenCV 5.0.0 的 HAL bug。
    #   BILINEAR 双线性插值，对图像质量友好。

    img = np.asarray(img_pil, dtype=np.float32) / 255.0
    # ↑ 转回 numpy 并归一化到 [0, 1]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # ↑ ImageNet 训练集统计的均值/标准差（RGB 通道）

    img = (img - mean) / std
    # ↑ 标准化：每个通道减去均值再除以标准差

    img = img.transpose(2, 0, 1)
    # ↑ HWC → CHW。PyTorch 要求通道维在前。

    tensor = torch.from_numpy(img).unsqueeze(0).float()
    # ↑ numpy → tensor；unsqueeze(0) 增加 batch 维 → [1, 3, H, W]

    return tensor


def infer_mask(model, frame_bgr, input_size, threshold, device, use_fp16=False):
    """
    对单帧做推理，返回与原始帧同尺寸的前景 alpha 掩码。

    参数:
        model       : 已加载的 U²-Net
        frame_bgr   : 原始 BGR 帧
        input_size  : 模型输入边长
        threshold   : 前景判定阈值（0~1）
        device      : torch.device
        use_fp16    : 是否使用 FP16

    返回:
        alpha : [H, W] float32，范围 0~1；出错返回 None
    """
    # ---- 空输入保护 ----
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    if h == 0 or w == 0:
        return None

    try:
        # ---- 预处理 + 搬到设备 ----
        tensor = preprocess_frame(frame_bgr, input_size).to(device, non_blocking=True)
        # ↑ non_blocking=True 允许 CPU→GPU 异步拷贝，速度略快

        with torch.no_grad():
            # ↑ 关闭梯度计算，节省显存、加快速度
            if use_fp16 and device.type == 'cuda':
                # FP16 需要 CUDA
                with torch.cuda.amp.autocast():
                    outputs = model(tensor)
            else:
                outputs = model(tensor)

        d0 = outputs[0]
        # ↑ 只取最终融合输出 d0

        if d0 is None or d0.numel() == 0:
            return None

        prob = torch.sigmoid(d0)[0, 0].float().cpu().numpy()
        # ↑ 拆解：
        #   torch.sigmoid(d0) : [1,1,H,W] 概率图
        #   [0, 0]            : 取 batch=0 通道=0 → [H, W]
        #   .float()          : 确保 float32（FP16 时是 float16）
        #   .cpu().numpy()    : 搬到 CPU 并转 numpy

        if prob.size == 0 or np.isnan(prob).any():
            return None
        # ↑ 检查无效输出（NaN 通常表示模型/权重有问题）

        # ---- 上采样回摄像头尺寸（用 PIL 避开 OpenCV 5.0.0 bug）----
        prob_u8 = (prob * 255.0).clip(0, 255).astype(np.uint8)
        # ↑ 转 uint8 更利于 PIL 处理，clip 防止浮点溢出

        prob_pil = Image.fromarray(prob_u8)
        prob_pil = prob_pil.resize((w, h), Image.BILINEAR)
        prob = np.asarray(prob_pil, dtype=np.float32) / 255.0

        # ---- 阈值二值化 ----
        mask = (prob >= threshold).astype(np.uint8) * 255
        # ↑ 大于阈值 = 前景 = 255；否则 = 0

        # ---- 形态学开运算去噪 ----
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # ↑ MORPH_OPEN = 先腐蚀后膨胀
        #   作用：去掉边缘零散的小噪点，保持主体形状

        # ---- 高斯模糊羽化边缘 ----
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0)
        # ↑ ksize=(0,0) 表示由 sigma 自动决定核大小
        #   sigmaX=2.0 越大越模糊，边缘过渡越柔和

        alpha = mask.astype(np.float32) / 255.0
        # ↑ 归一化到 0~1，方便后续 Alpha 合成

        return alpha

    except Exception as e:
        print(f"推理内部错误: {e}")
        traceback.print_exc()
        return None
        # ↑ 任何异常都返回 None，让上层做降级处理，程序不崩溃


def composite(frame_bgr, alpha, background_bgr):
    """
    Alpha 合成：Output = alpha * 前景 + (1-alpha) * 背景

    参数:
        frame_bgr      : 原始摄像头帧 [H, W, 3] BGR
        alpha          : 前景掩码 [H, W] float32，0~1
        background_bgr : 背景图 [H, W, 3] BGR，尺寸与前景相同

    返回:
        output : [H, W, 3] uint8 BGR
    """
    a = alpha[..., None]
    # ↑ 把 [H, W] 扩展成 [H, W, 1]，以便与三通道广播相乘

    return (frame_bgr.astype(np.float32) * a
            + background_bgr.astype(np.float32) * (1 - a)).astype(np.uint8)
    # ↑ 先转 float32 避免溢出，最后转回 uint8


def create_video_writer(path, fps, frame_size):
    """
    根据文件扩展名选择合适的编码器创建 VideoWriter。

    参数:
        path       : 输出文件路径
        fps        : 帧率
        frame_size : (width, height)

    返回:
        VideoWriter 或 None
    """
    fourcc_list = [
        ('mp4v', '.mp4'),   # MP4 容器，兼容性最好
        ('XVID', '.avi'),   # AVI + Xvid
        ('MJPG', '.avi'),   # AVI + Motion JPEG
    ]
    for fourcc_str, ext in fourcc_list:
        if not path.lower().endswith(ext):
            continue
        # ↑ 只尝试与文件扩展名匹配的编码器

        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        # ↑ 'mp4v' → 四个字符 'm','p','4','v' → 合成一个 32 位整数

        writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        if writer.isOpened():
            return writer
        # ↑ 打开成功就返回
        writer.release()
        # ↑ 打开失败要释放，否则资源泄漏

    # 都失败则用默认 mp4v 再试一次
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
    return writer if writer.isOpened() else None


def find_ffmpeg(user_path=None):
    """
    在多个位置查找 ffmpeg 可执行文件。

    查找顺序:
        1) 用户指定路径
        2) 系统 PATH
        3) 常见默认安装位置

    返回:
        找到的路径字符串，或 None
    """
    if user_path and os.path.isfile(user_path):
        return user_path

    p = shutil.which("ffmpeg")
    if p:
        return p

    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"D:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    return None


# ==============================================================================
# 【模块 7：音频录制类】
# ==============================================================================

class AudioRecorder:
    """
    使用 sounddevice 采集麦克风音频。

    工作方式:
        - 创建 InputStream 时传入 callback 函数
        - sounddevice 在独立线程中每拿到一块音频就调用 callback
        - callback 把数据 append 到 self.frames
        - stop() 时把所有块拼接，写成 16-bit 单声道 WAV

    注意:
        - callback 运行在音频线程，必须加锁保护 frames
        - 用 float32 采集，转 int16 时用 *32767 而不是 *32768，
          因为 int16 的正最大是 32767，乘 32768 会溢出
    """
    def __init__(self, samplerate=44100, channels=1):
        self.samplerate = samplerate
        # ↑ 采样率：44100 Hz 是标准 CD 音质
        self.channels = channels
        # ↑ 声道数：1=单声道，2=立体声
        self.frames = []
        # ↑ 存放每块音频数据的列表（numpy 数组）
        self.stream = None
        self.recording = False
        self.lock = threading.Lock()
        # ↑ 互斥锁：保护 frames 的读写

    def start(self, device=None):
        """启动麦克风采集。device=None 表示系统默认设备。"""
        if not HAS_SOUNDDEVICE:
            raise RuntimeError("未安装 sounddevice，无法录音。请 pip install sounddevice")

        self.frames = []
        self.recording = True

        def callback(indata, frames, time_info, status):
            """
            sounddevice 回调函数。
            参数:
                indata   : [frames, channels] float32 数组，本次采集的音频块
                frames   : 本块包含的采样点数
                time_info: 时间戳信息（本程序不用）
                status   : 状态标志（overflow 等）
            """
            if status:
                print(f"音频回调状态: {status}")

            with self.lock:
                # ↑ 加锁，防止与 stop() 竞争
                if self.recording:
                    self.frames.append(indata.copy())
                    # ↑ 必须 copy，否则下一帧会覆盖同一内存

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            device=device,
            callback=callback
        )
        self.stream.start()

    def stop(self, wav_path):
        """停止采集并保存为 WAV。返回 True 表示成功。"""
        self.recording = False

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        with self.lock:
            if not self.frames:
                return False
            # ↑ 没有采到任何数据，无法生成文件

            audio_data = np.concatenate(self.frames, axis=0)
            # ↑ 沿时间轴把所有块拼成一个 [total_samples, channels] 数组

            audio_int16 = np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16)
            # ↑ float32 [-1,1] → int16 [-32768,32767]
            #   clip 防止因浮点误差导致溢出

            with wave.open(wav_path, 'wb') as wf:
                # ↑ 'wb' = 写二进制模式
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                # ↑ 采样位宽 2 字节 = 16 bit
                wf.setframerate(self.samplerate)
                wf.writeframes(audio_int16.tobytes())
                # ↑ tobytes() 把 numpy 数组转成原始字节流

        return True


# ==============================================================================
# 【模块 8：视频线程】
# ==============================================================================

class VideoThread(QThread):
    """
    后台线程：循环读帧 → 推理 → 合成 → 通过信号发给 GUI。

    三个自定义信号:
        frame_ready(np.ndarray) : 一帧合成好的 BGR 画面
        status_update(str)      : 状态栏文字
        error_signal(str)       : 错误信息

    Qt 信号槽的跨线程通信是线程安全的：
        子线程 emit 信号，槽函数在 GUI 线程执行。
    """
    frame_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, camera_id, model_path, device_str, input_size,
                 threshold, mirror, bg_mode, bg_path, bg_color,
                 infer_interval=2, use_fp16=True):
        super().__init__()
        self.camera_id = camera_id
        self.model_path = model_path
        self.device_str = device_str
        self.input_size = input_size
        self.threshold = threshold
        self.mirror = mirror
        self.bg_mode = bg_mode
        self.bg_path = bg_path
        self.bg_color = bg_color
        self.infer_interval = infer_interval
        self.use_fp16 = use_fp16
        self.running = False

        self._cached_bg_path = None
        self._cached_bg_image = None
        # ↑ 缓存背景图，避免每帧都读文件

    def get_background(self, frame_shape):
        """
        返回与摄像头帧同尺寸的背景图（BGR）。
        纯色模式直接生成；图片模式从缓存读取或加载后缩放。
        """
        h, w = frame_shape[:2]

        if self.bg_mode == "color":
            return np.full((h, w, 3), self.bg_color, dtype=np.uint8)
            # ↑ np.full 用同一个值填充整个数组；
            #   形状 (h, w, 3) 与 BGR 图像对应；
            #   值 self.bg_color 是 (B, G, R) 元组

        # ---- 图片模式：缓存机制 ----
        if self.bg_path != self._cached_bg_path or self._cached_bg_image is None:
            # ↑ 只有路径变化或首次加载时才读文件
            try:
                img = Image.open(self.bg_path).convert("RGB")
                # ↑ PIL 打开并强制转 RGB（处理灰度图/带 alpha 的 PNG）
                bg = np.array(img)
                bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
                # ↑ 转 BGR 与 OpenCV 保持一致
                self._cached_bg_image = bg
                self._cached_bg_path = self.bg_path
            except Exception as e:
                self.error_signal.emit(f"背景图片加载失败: {e}")
                return np.full((h, w, 3), (255, 0, 0), dtype=np.uint8)
                # ↑ 失败时用纯蓝色兜底

        # ---- 尺寸适配 ----
        if self._cached_bg_image.shape[0] != h or self._cached_bg_image.shape[1] != w:
            bg_pil = Image.fromarray(cv2.cvtColor(self._cached_bg_image,
                                                  cv2.COLOR_BGR2RGB))
            bg_pil = bg_pil.resize((w, h), Image.BILINEAR)
            bg = cv2.cvtColor(np.asarray(bg_pil), cv2.COLOR_RGB2BGR)
            return bg
        return self._cached_bg_image

    def run(self):
        """
        线程主体。
        执行顺序:
            1. 选择设备
            2. 加载模型
            3. 打开摄像头
            4. 循环：读帧 → 镜像 → 隔帧推理 → 合成 → 发信号
            5. finally 释放摄像头
        """
        self.running = True
        cap = None
        try:
            # ============ 选择设备 ============
            if self.device_str == "cuda" and torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
                if self.device_str == "cuda":
                    self.status_update.emit("CUDA 不可用，已回退 CPU")

            # ============ 加载模型 ============
            self.status_update.emit("正在加载 U2-Net 模型...")
            model = load_u2net_model(self.model_path, device)
            self.status_update.emit("模型加载完成，正在打开摄像头...")

            # ============ 打开摄像头 ============
            cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            # ↑ Windows 下 CAP_DSHOW 比默认后端更快
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # ↑ 缓冲区设为 1，避免历史帧积压导致画面延迟

            if not cap.isOpened():
                self.error_signal.emit(f"无法打开摄像头 {self.camera_id}，可能被占用或不存在")
                return

            self.status_update.emit("摄像头已打开，开始实时虚拟背景")

            # ============ 循环变量 ============
            frame_count = 0
            last_mask = None
            prev_time = time.time()
            fps = 0.0
            infer_ms = 0.0

            # ============ 主循环 ============
            while self.running:
                ret, frame = cap.read()

                # 关键：摄像头初始化阶段可能返回 True 但 frame 是空数组
                if not ret or frame is None or frame.size == 0:
                    time.sleep(0.005)
                    continue
                    # ↑ 让出 CPU 时间片，避免忙等

                if self.mirror:
                    frame = cv2.flip(frame, 1)
                    # ↑ flipCode=1 表示水平翻转

                # ---- 隔帧推理：每 N 帧才跑一次模型 ----
                if frame_count % self.infer_interval == 0 or last_mask is None:
                    t0 = time.time()
                    new_mask = infer_mask(
                        model, frame, self.input_size,
                        self.threshold, device, self.use_fp16
                    )
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                        # ↑ 等待 GPU 真正完成计算，否则计时不准
                    infer_ms = (time.time() - t0) * 1000.0
                    if new_mask is not None:
                        last_mask = new_mask
                    # ↑ 若本次失败，保留上一次的掩码

                # ---- 合成 ----
                if last_mask is None:
                    output = frame
                    # ↑ 首次推理失败时，直接显示原图
                else:
                    bg = self.get_background(frame.shape)
                    output = composite(frame, last_mask, bg)

                # ---- FPS 统计 ----
                frame_count += 1
                now = time.time()
                if now - prev_time >= 1.0:
                    fps = frame_count / (now - prev_time)
                    frame_count = 0
                    prev_time = now
                    # ↑ 每秒更新一次 FPS

                # ---- 发信号给 GUI ----
                self.frame_ready.emit(output)
                self.status_update.emit(
                    f"FPS: {fps:.1f} | 推理: {infer_ms:.1f} ms | "
                    f"设备: {device.type} | 输入: {self.input_size}"
                )

        except Exception as e:
            self.error_signal.emit(f"运行错误: {e}\n{traceback.format_exc()}")
        finally:
            if cap is not None:
                cap.release()
            self.status_update.emit("已停止")

    def stop(self):
        """请求线程停止，最多等 3 秒。"""
        self.running = False
        self.wait(3000)
        # ↑ wait(ms) 阻塞当前线程直到目标线程结束或超时


# ==============================================================================
# 【模块 9：主窗口】
# ==============================================================================

class VirtualBackgroundApp(QMainWindow):
    """
    主窗口类。
    职责:
        - 构建界面
        - 响应用户操作
        - 管理 VideoThread 生命周期
        - 处理录制逻辑
    """

    # 连接诊断在后台线程里跑（ping 和 TCP 探测都会阻塞数秒），
    # 结果通过这个信号回到 GUI 线程显示，避免界面卡住。
    diagnostic_finished = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("U2-Net 虚拟背景与局域网会议 - PyQt5")
        self.resize(1500, 900)

        # ---- 运行时状态 ----
        self.thread = None
        self.current_frame = None
        self.bg_color = (255, 0, 0)
        # ↑ 默认蓝色（BGR 顺序！不是 RGB）

        # ---- 录制状态 ----
        self.recording = False
        self.video_writer = None
        self.record_path = None
        self.final_path = None
        self.audio_path = None
        self.record_fps = 20
        self.audio_recorder = None

        # ---- 局域网会议状态 ----
        # 网络线程只能通过 Qt 信号把数据送回 GUI 线程，不能直接操作界面。
        self.conference = LANConference(self)
        self.conference.remote_frame.connect(self.display_remote_frame)
        self.conference.mosaic_frame.connect(self.display_mosaic)
        self.conference.remote_audio.connect(self.play_remote_audio)
        self.conference.members_changed.connect(self.update_conference_members)
        self.conference.status.connect(self.update_conference_status)
        self.conference.error.connect(self.show_conference_error)
        self.remote_tiles = {}
        self.remote_frames = {}
        self.member_order = []
        # ↑ 成员顺序（含自己）。合成模式下靠它把大图里的格子对应回具体的人，
        #   顺序必须和主机合成时用的完全一致。
        self.diagnostic_running = False
        self.diagnostic_finished.connect(self.show_diagnostic_result)

        # ---- 会议语音状态 ----
        # 进会即打开音频设备（默认闭麦），因此 Speaker 一直在播放他人语音，
        # 是否需要说话由“开麦/闭麦”决定。
        self.audio_link = None

        self.init_ui()
        self.check_environment()

    # ---------------- UI 构建 ----------------

    def init_ui(self):
        """构建整个界面：左侧控制面板，右侧视频预览。"""
        central = QWidget()
        self.setCentralWidget(central)
        # ↑ QMainWindow 需要设置 centralWidget 才能有内容

        main_layout = QHBoxLayout(central)
        # ↑ 主布局：水平排列（左边控制面板 + 右边视频显示）

        # ============ 左侧面板 ============
        # 控件越加越多，套一层滚动区：窗口不够高时可以滚动查看，而不是把底部按钮裁掉。
        panel = QWidget()
        panel.setMinimumWidth(360)
        panel_layout = QVBoxLayout(panel)

        # ---- 摄像头组 ----
        cam_group = QGroupBox("摄像头")
        cam_layout = QFormLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["0", "1", "2", "3"])
        self.camera_combo.setCurrentText("0")
        cam_layout.addRow("摄像头编号:", self.camera_combo)
        cam_group.setLayout(cam_layout)
        panel_layout.addWidget(cam_group)

        # ---- 模型组 ----
        model_group = QGroupBox("模型")
        model_layout = QFormLayout()

        self.model_edit = QLineEdit("u2net_human_seg.pth")
        model_btn = QPushButton("浏览...")
        model_btn.clicked.connect(self.browse_model)
        # ↑ Qt 信号槽：按钮点击 → 调用 browse_model

        model_h = QHBoxLayout()
        model_h.addWidget(self.model_edit)
        model_h.addWidget(model_btn)
        model_layout.addRow("权重路径:", model_h)
        # ↑ QFormLayout.addRow 第二个参数可以是控件或布局

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        if not torch.cuda.is_available():
            self.device_combo.setCurrentText("cpu")
        model_layout.addRow("设备:", self.device_combo)

        self.size_combo = QComboBox()
        self.size_combo.addItems(["192", "256", "320", "384"])
        self.size_combo.setCurrentText("320")
        model_layout.addRow("输入尺寸:", self.size_combo)

        self.fp16_check = QCheckBox("FP16 加速 (CUDA)")
        self.fp16_check.setChecked(True)
        model_layout.addRow(self.fp16_check)

        model_group.setLayout(model_layout)
        panel_layout.addWidget(model_group)

        # ---- 背景组 ----
        bg_group = QGroupBox("背景")
        bg_layout = QVBoxLayout()

        self.bg_image_radio = QRadioButton("图片背景")
        self.bg_color_radio = QRadioButton("纯色背景")
        self.bg_image_radio.setChecked(True)

        self.bg_group = QButtonGroup()
        self.bg_group.addButton(self.bg_image_radio)
        self.bg_group.addButton(self.bg_color_radio)
        # ↑ QButtonGroup 保证两个单选按钮互斥

        bg_layout.addWidget(self.bg_image_radio)

        bg_path_h = QHBoxLayout()
        self.bg_edit = QLineEdit()
        bg_btn = QPushButton("浏览...")
        bg_btn.clicked.connect(self.browse_background)
        bg_path_h.addWidget(self.bg_edit)
        bg_path_h.addWidget(bg_btn)
        bg_layout.addLayout(bg_path_h)

        bg_layout.addWidget(self.bg_color_radio)
        self.color_btn = QPushButton("选择纯色颜色")
        self.color_btn.clicked.connect(self.choose_color)
        self.color_label = QLabel("当前颜色: 蓝色")
        bg_layout.addWidget(self.color_btn)
        bg_layout.addWidget(self.color_label)

        bg_group.setLayout(bg_layout)
        panel_layout.addWidget(bg_group)

        # ---- 参数组 ----
        param_group = QGroupBox("参数")
        param_layout = QFormLayout()

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)
        self.threshold_label = QLabel("0.50")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(f"{v/100:.2f}")
        )
        # ↑ 匿名 lambda：滑块值变化时立即更新数值标签

        threshold_h = QHBoxLayout()
        threshold_h.addWidget(self.threshold_slider)
        threshold_h.addWidget(self.threshold_label)
        param_layout.addRow("阈值:", threshold_h)

        self.mirror_check = QCheckBox("水平镜像")
        self.mirror_check.setChecked(True)
        param_layout.addRow(self.mirror_check)

        param_group.setLayout(param_layout)
        panel_layout.addWidget(param_group)

        # ---- 语音 / 声音组 ----
        audio_group = QGroupBox("语音 / 声音")
        audio_layout = QFormLayout()

        self.mic_combo = QComboBox()
        self.speaker_combo = QComboBox()
        self.refresh_audio_devices()
        # ↑ 麦克风同时供会议语音和本地录制使用，避免界面上出现两个麦克风下拉框。

        mic_h = QHBoxLayout()
        mic_h.addWidget(self.mic_combo)
        refresh_audio_btn = QPushButton("刷新")
        refresh_audio_btn.clicked.connect(self.refresh_audio_devices)
        mic_h.addWidget(refresh_audio_btn)
        audio_layout.addRow("麦克风:", mic_h)

        speaker_h = QHBoxLayout()
        speaker_h.addWidget(self.speaker_combo)
        refresh_speaker_btn = QPushButton("刷新")
        refresh_speaker_btn.clicked.connect(self.refresh_audio_devices)
        speaker_h.addWidget(refresh_speaker_btn)
        audio_layout.addRow("扬声器:", speaker_h)

        self.voice_btn = QPushButton("开麦")
        self.voice_btn.setCheckable(True)
        self.voice_btn.setChecked(False)
        self.voice_btn.setEnabled(False)
        # ↑ 默认闭麦：进会先不收音，免得一进去就泄露环境音或引起啸叫。
        self.voice_btn.toggled.connect(self.toggle_voice)
        audio_layout.addRow(self.voice_btn)

        self.voice_state_label = QLabel("语音：未加入会议")
        audio_layout.addRow(self.voice_state_label)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_label = QLabel("100%")
        volume_h = QHBoxLayout()
        volume_h.addWidget(self.volume_slider)
        volume_h.addWidget(self.volume_label)
        audio_layout.addRow("输出音量:", volume_h)
        self.volume_slider.valueChanged.connect(self.update_volume)

        self.record_audio_check = QCheckBox("录制时包含声音")
        self.record_audio_check.setChecked(True)
        audio_layout.addRow(self.record_audio_check)

        self.ffmpeg_edit = QLineEdit()
        ffmpeg_btn = QPushButton("浏览...")
        ffmpeg_btn.clicked.connect(self.browse_ffmpeg)

        ffmpeg_h = QHBoxLayout()
        ffmpeg_h.addWidget(self.ffmpeg_edit)
        ffmpeg_h.addWidget(ffmpeg_btn)
        audio_layout.addRow("ffmpeg 路径:", ffmpeg_h)

        audio_group.setLayout(audio_layout)
        panel_layout.addWidget(audio_group)

        # ---- 局域网会议组 ----
        # 一台电脑创建主机后成为转发节点；其他电脑填写该电脑的局域网 IP 加入。
        meeting_group = QGroupBox("局域网会议")
        meeting_layout = QFormLayout()

        self.meeting_name_edit = QLineEdit("参会者")
        meeting_layout.addRow("昵称:", self.meeting_name_edit)

        self.meeting_host_edit = QLineEdit(self.get_local_ip())
        self.meeting_host_edit.setPlaceholderText("主机局域网 IP，例如 192.168.1.20")
        meeting_layout.addRow("主机 IP:", self.meeting_host_edit)

        self.meeting_port_edit = QLineEdit("49500")
        meeting_layout.addRow("端口:", self.meeting_port_edit)

        self.meeting_capacity_spin = QSpinBox()
        self.meeting_capacity_spin.setRange(2, MAX_PARTICIPANTS)
        self.meeting_capacity_spin.setValue(18)
        # ↑ 默认仍留 18：上限放到 50 是允许你按需调大，但人数越多，主机要转发的视频
        #   路数增长得越快（近似平方），默认值取一个普通局域网还扛得住的数字。
        self.meeting_capacity_spin.setToolTip(
            f"本机作为主机时允许的总人数（含主持人），最大 {MAX_PARTICIPANTS}。\n"
            "只有“创建主机”时生效。\n\n"
            "注意：人数越多，主机需要转发的视频路数增长越快，"
            "实际能开多少人取决于网络带宽。"
        )
        meeting_layout.addRow("人数上限:", self.meeting_capacity_spin)

        self.meeting_mode_combo = QComboBox()
        self.meeting_mode_combo.addItem("合成画面（人多时用）", MOSAIC_MODE)
        self.meeting_mode_combo.addItem("逐路转发（画质好）", RELAY)
        self.meeting_mode_combo.setToolTip(
            "合成画面：主机把所有人的画面拼成一张多宫格图，只发一路。\n"
            "  主机上行带宽随人数线性增长，几十人也扛得住；\n"
            "  代价是所有人看到的是同一张缩略图墙，格子会随人数变小。\n\n"
            "逐路转发：每个人收到其他人的原始 480×270 画面，单人格子最大最清晰；\n"
            "  但主机上行带宽随人数近似平方增长，十几人就会开始卡。\n\n"
            "只有“创建主机”时生效；加入别人的会议时由主机决定，不需要这边设置一致。"
        )
        meeting_layout.addRow("画面模式:", self.meeting_mode_combo)

        self.meeting_fps_combo = QComboBox()
        self.meeting_fps_combo.addItems(["8", "10", "12", "15"])
        self.meeting_fps_combo.setCurrentText("12")
        meeting_layout.addRow("发送帧率:", self.meeting_fps_combo)

        self.meeting_quality_slider = QSlider(Qt.Horizontal)
        self.meeting_quality_slider.setRange(30, 85)
        self.meeting_quality_slider.setValue(55)
        self.meeting_quality_label = QLabel("55")
        self.meeting_quality_slider.valueChanged.connect(
            lambda value: self.meeting_quality_label.setText(str(value))
        )
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(self.meeting_quality_slider)
        quality_layout.addWidget(self.meeting_quality_label)
        meeting_layout.addRow("JPEG 质量:", quality_layout)

        meeting_buttons = QHBoxLayout()
        self.host_meeting_btn = QPushButton("创建主机")
        self.host_meeting_btn.clicked.connect(self.start_conference_host)
        self.join_meeting_btn = QPushButton("加入会议")
        self.join_meeting_btn.clicked.connect(self.join_conference)
        self.leave_meeting_btn = QPushButton("离开")
        self.leave_meeting_btn.setEnabled(False)
        self.leave_meeting_btn.clicked.connect(self.leave_conference)
        meeting_buttons.addWidget(self.host_meeting_btn)
        meeting_buttons.addWidget(self.join_meeting_btn)
        meeting_buttons.addWidget(self.leave_meeting_btn)
        meeting_layout.addRow(meeting_buttons)

        self.meeting_state_label = QLabel("未加入会议")
        self.meeting_state_label.setWordWrap(True)
        meeting_layout.addRow(self.meeting_state_label)

        # 连不上时先点这里：一次性报出本机 IP、监听状态、目标可达性与端口连通性，
        # 用来区分「程序没起来」「填错 IP」「防火墙拦截」「网络客户端隔离」。
        self.diag_btn = QPushButton("连接诊断")
        self.diag_btn.clicked.connect(self.run_connection_diagnostic)
        meeting_layout.addRow(self.diag_btn)

        meeting_group.setLayout(meeting_layout)
        panel_layout.addWidget(meeting_group)

        # ---- 保存组 ----
        save_group = QGroupBox("保存")
        save_layout = QFormLayout()

        self.save_dir_edit = QLineEdit("outputs")
        save_dir_btn = QPushButton("浏览...")
        save_dir_btn.clicked.connect(self.browse_save_dir)

        save_dir_h = QHBoxLayout()
        save_dir_h.addWidget(self.save_dir_edit)
        save_dir_h.addWidget(save_dir_btn)
        save_layout.addRow("保存目录:", save_dir_h)

        save_group.setLayout(save_layout)
        panel_layout.addWidget(save_group)

        # ---- 按钮区 ----
        # 第一行：开始 / 结束
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始")
        self.start_btn.clicked.connect(self.start_video)
        self.stop_btn = QPushButton("结束")
        self.stop_btn.setEnabled(False)
        # ↑ 初始禁用，只有当视频运行中才启用
        self.stop_btn.clicked.connect(self.stop_video)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        panel_layout.addLayout(btn_layout)

        # 第二行：保存 / 开始录制 / 停止录制
        btn_layout2 = QHBoxLayout()
        self.save_btn = QPushButton("截图")
        self.save_btn.clicked.connect(self.save_image)

        self.record_btn = QPushButton("开始录制")
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self.start_recording)

        self.stop_record_btn = QPushButton("停止录制")
        self.stop_record_btn.setEnabled(False)
        self.stop_record_btn.clicked.connect(self.stop_recording)

        btn_layout2.addWidget(self.save_btn)
        btn_layout2.addWidget(self.record_btn)
        btn_layout2.addWidget(self.stop_record_btn)
        panel_layout.addLayout(btn_layout2)

        # 第三行：退出
        btn_layout3 = QHBoxLayout()
        self.exit_btn = QPushButton("退出")
        self.exit_btn.clicked.connect(self.close)
        btn_layout3.addWidget(self.exit_btn)
        panel_layout.addLayout(btn_layout3)

        panel_layout.addStretch()
        # ↑ 弹簧：把前面的控件往上顶，避免均匀分布

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        # ↑ 让 panel 随滚动区宽度伸缩；否则 panel 会被压到最小尺寸
        scroll.setFixedWidth(396)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        main_layout.addWidget(scroll)

        # ============ 右侧视频显示与会议成员网格 ============
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.video_label = QLabel("摄像头预览")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        right_layout.addWidget(self.video_label, stretch=3)

        self.remote_group = QGroupBox("会议成员（远端画面）")
        remote_outer = QVBoxLayout(self.remote_group)
        remote_outer.setContentsMargins(6, 6, 6, 6)

        # 人数上限放宽后，满员时这张网格可能有八九行。套一层滚动区，让它自己滚，
        # 而不是把上面的本地预览挤扁。
        self.remote_scroll = QScrollArea()
        self.remote_scroll.setWidgetResizable(True)
        self.remote_scroll.setFrameShape(QFrame.NoFrame)
        remote_container = QWidget()
        self.remote_layout = QGridLayout(remote_container)
        self.remote_layout.setSpacing(6)
        for column in range(REMOTE_GRID_COLUMNS):
            self.remote_layout.setColumnStretch(column, 1)
        # ↑ 让各列均分宽度，格子大小一致，而不是各自缩到最小尺寸
        self.remote_scroll.setWidget(remote_container)
        remote_outer.addWidget(self.remote_scroll)

        self.remote_empty_label = QLabel("创建或加入局域网会议后，远端成员画面会显示在这里")
        self.remote_empty_label.setAlignment(Qt.AlignCenter)
        self.remote_layout.addWidget(self.remote_empty_label, 0, 0, 1, REMOTE_GRID_COLUMNS)
        right_layout.addWidget(self.remote_group, stretch=2)
        main_layout.addWidget(right_panel, stretch=1)
        # ↑ 本地预览在上；远端成员以 REMOTE_GRID_COLUMNS 列的网格显示在下，超出部分可滚动。

        # ============ 状态栏 ============
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        # 参数变化实时生效
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.mirror_check.stateChanged.connect(self.update_mirror)

    # ---------------- 局域网会议 ----------------

    @staticmethod
    def get_local_ip():
        """取得最可能可被局域网设备访问的 IPv4 地址。

        UDP connect 不会真的向外发包，只是让系统根据路由表选择出口网卡；无网络时
        回退到 127.0.0.1，仍可用于同一台机器上的多开测试。
        """
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"
        finally:
            probe.close()

    def _conference_settings(self):
        """读取并校验会议参数，同时将压缩策略写入传输控制器。"""
        try:
            port = int(self.meeting_port_edit.text().strip())
            if not 1024 <= port <= 65535:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "会议设置", "端口必须是 1024–65535 之间的整数")
            return None
        self.conference.fps = int(self.meeting_fps_combo.currentText())
        self.conference.quality = self.meeting_quality_slider.value()
        return port

    def _set_conference_buttons(self, active):
        self.host_meeting_btn.setEnabled(not active)
        self.join_meeting_btn.setEnabled(not active)
        self.leave_meeting_btn.setEnabled(active)

    def start_conference_host(self):
        """在本机监听会议端口；总人数上限由界面上的“人数上限”决定（主机也计入）。"""
        port = self._conference_settings()
        if port is None:
            return
        self.conference.mode = self.meeting_mode_combo.currentData()
        # ↑ 画面模式只在创建主机时有意义，加入别人的会议时由对方决定。
        self.conference.host(
            self.meeting_name_edit.text(), port,
            max_participants=self.meeting_capacity_spin.value()
        )
        if self.conference.connected:
            self.meeting_host_edit.setText(self.get_local_ip())
            self._set_conference_buttons(True)
            self._open_voice_link()
            self.update_conference_members(self.conference.server.members())

    def join_conference(self):
        """连接局域网主机。客户端只上传一条压缩后的本地视频流与一路语音。"""
        port = self._conference_settings()
        host = self.meeting_host_edit.text().strip()
        if port is None:
            return
        if not host:
            QMessageBox.warning(self, "会议设置", "请输入会议主机的局域网 IP")
            return
        self.conference.join(host, port, self.meeting_name_edit.text())
        if self.conference.connected:
            self._set_conference_buttons(True)
            self._open_voice_link()

    def leave_conference(self):
        """断开网络并清理远端成员网格；不会停止本地虚拟背景预览。"""
        self._close_voice_link()
        self.conference.leave()
        self.remote_frames.clear()
        self.member_order = []
        self._clear_remote_grid()
        self._set_conference_buttons(False)
        self.meeting_state_label.setText("未加入会议")

    # ---------------- 会议语音 ----------------

    def _open_voice_link(self):
        """进会时打开麦克风与扬声器。

        默认闭麦：扬声器立刻开始播放他人语音，但麦克风一块都不上传，免得一进会就
        把环境音送出去，或者和外放形成啸叫。
        """
        if self.audio_link is not None:
            return
        if not HAS_SOUNDDEVICE:
            self.voice_state_label.setText("语音：未安装 sounddevice，只能传画面")
            return
        try:
            link = AudioLink(on_capture=self.conference.publish_audio)
            link.set_muted(True)
            link.set_volume(self.volume_slider.value() / 100.0)
            link.start(self.mic_combo.currentData(), self.speaker_combo.currentData())
        except Exception as e:
            self.voice_state_label.setText("语音：音频设备打开失败")
            QMessageBox.warning(self, "会议语音", f"无法打开音频设备：{e}")
            return
        self.audio_link = link
        self.voice_btn.setEnabled(True)
        self.voice_state_label.setText("语音：已闭麦（点“开麦”才能说话）")

    def _close_voice_link(self):
        """关掉音频流并复位按钮。没开会时调用也安全。"""
        link, self.audio_link = self.audio_link, None
        if link is not None:
            link.stop()
        self.voice_btn.blockSignals(True)
        # ↑ 复位会触发 toggled，这里先挡掉，避免回调再去碰已经关掉的 audio_link。
        self.voice_btn.setChecked(False)
        self.voice_btn.blockSignals(False)
        self.voice_btn.setText("开麦")
        self.voice_btn.setEnabled(False)
        self.voice_state_label.setText("语音：未加入会议")

    def toggle_voice(self, speaking):
        """开麦/闭麦只决定是否上传麦克风；扬声器始终在播放他人语音。"""
        if self.audio_link is None:
            return
        self.audio_link.set_muted(not speaking)
        self.voice_btn.setText("闭麦" if speaking else "开麦")
        self.voice_state_label.setText("语音：已开麦" if speaking else "语音：已闭麦")

    def update_volume(self, value):
        self.volume_label.setText(f"{value}%")
        if self.audio_link is not None:
            self.audio_link.set_volume(value / 100.0)

    def play_remote_audio(self, participant_id, name, pcm):
        """信号已经把远端语音送回主线程，这里只负责塞进对应说话人的播放缓冲。"""
        if self.audio_link is not None:
            self.audio_link.push_remote(participant_id, pcm)

    def update_conference_status(self, message):
        self.meeting_state_label.setText(message)
        if self.audio_link is not None and not self.conference.connected:
            # 主机掉线时网络层只发状态、不抛“错误”，但语音设备得跟着收回来，
            # 否则麦克风会一直开着，对着一个已经断掉的连接空发。
            self._close_voice_link()
            self._set_conference_buttons(False)
            # ↑ host() 启动主机时也会先发一条状态，但那时 audio_link 还是 None，
            #   所以这条清理只会在真正的掉线时执行。

    def show_conference_error(self, message):
        """会议错误只影响网络层，不能调用 show_error() 以免误停本地摄像头。"""
        self.meeting_state_label.setText(message)
        self._set_conference_buttons(False)
        QMessageBox.warning(self, "局域网会议", message)

    # ---------------- 连接诊断 ----------------

    def run_connection_diagnostic(self):
        """启动后台诊断线程。本函数只负责发起，探测耗时都在线程里。"""
        if self.diagnostic_running:
            return
        self.diagnostic_running = True
        self.diag_btn.setEnabled(False)
        self.status_bar.showMessage("正在诊断连接，请稍候…")

        target = self.meeting_host_edit.text().strip()
        try:
            port = int(self.meeting_port_edit.text().strip())
        except ValueError:
            port = 49500
        # ↑ 端口不合法也给个默认值继续诊断，因为诊断本身就是要帮用户排查填错的情况。

        threading.Thread(target=self._diagnose, args=(target, port), daemon=True).start()

    def _diagnose(self, target, port):
        """后台线程主体：只做网络探测，绝不触碰任何 Qt 控件。"""
        lines = []
        try:
            local_ip = self.get_local_ip()
            lines.append(f"① 本机局域网 IP：{local_ip}")
            lines.append("    其他成员要填这个地址，不要填 127.0.0.1，也不要填虚拟网卡地址。")

            if self.conference.is_host:
                # 必须报告主机真正绑定的端口，而不是输入框里的值——
                # 用户改过端口输入框后，两者可能不一致。
                listen_port = self.conference.server.port
                lines.append(f"② 本机端口监听：正常（0.0.0.0:{listen_port}）")
                if listen_port != port:
                    lines.append(f"    注意：输入框填的是 {port}，与主机实际监听的 "
                                 f"{listen_port} 不一致，其他成员必须填 {listen_port}。")
            else:
                listen_port = None
                if self.conference.connected:
                    lines.append("② 本机端口监听：无（当前是参会者，不需要监听）")
                else:
                    lines.append("② 本机端口监听：未开启（还没点“创建主机”）")

            self_target = target in (local_ip, "127.0.0.1", "localhost")

            if not target:
                lines.append("③ 目标检测：已跳过（“主机 IP”一栏为空）")
                lines.append("    想检测对方，请先在“主机 IP”里填写对方地址。")
                conclusion = "主机端：点“创建主机”后，把上面的本机 IP 和端口报给其他成员。"
            elif self_target:
                if listen_port is not None:
                    if port == listen_port:
                        lines.append(f"③ 目标 {target} 就是本机，监听已开启，跳过重复探测。")
                        conclusion = "主机已就绪：把上面的本机 IP 和端口报给其他成员即可。"
                    else:
                        lines.append(f"③ 目标 {target} 是本机地址，但端口对不上：")
                        lines.append(f"    主机实际监听 {listen_port}，而输入框填的是 {port}。")
                        conclusion = (f"端口填错了；其他成员应当连接 "
                                      f"{local_ip}:{listen_port}。")
                else:
                    # 本机没在开会，但仍要探测：同一台机器多开时确实可以连 127.0.0.1。
                    port_ok, detail = self._probe_port(target, port)
                    if port_ok:
                        lines.append(f"③ 目标 {target}:{port} 连通，本机上有服务在监听。")
                        conclusion = ("连通：可能是同一台机器上多开的另一个实例。"
                                      "若对方在别的电脑上，请把 IP 换成对方显示的那个。")
                    else:
                        lines.append(f"③ 目标 {target} 是本机地址，该端口无人监听。")
                        lines.append(f"    底层错误：{detail}")
                        conclusion = ("你填的是自己的地址；若对方在另一台电脑上，"
                                      "请填对方主机的局域网 IP。")
            else:
                ping_ok = self._ping(target)
                lines.append(f"③ 能否 ping 通 {target}：{'是' if ping_ok else '否'}")
                port_ok, detail = self._probe_port(target, port)
                if port_ok:
                    lines.append(f"④ 能否连上 {target}:{port}：是")
                    conclusion = "网络与端口都通；若仍连不上，请确认双方端口号完全一致。"
                else:
                    lines.append(f"④ 能否连上 {target}:{port}：否")
                    lines.append(f"    底层错误：{detail}")
                    if ping_ok:
                        conclusion = ("能 ping 通但端口连不上：对方多半不在“创建主机”状态；"
                                      "若确认已创建，则是对方防火墙拦住了该端口。")
                    else:
                        conclusion = ("完全不通：两台设备可能不在同一网络，或该网络开启了"
                                      "客户端隔离（常见于手机热点、校园/单位 Wi-Fi）。")
        except Exception as exc:
            lines.append(f"诊断过程出错：{exc}")
            conclusion = "诊断未完成，请重试。"

        lines.append("")
        lines.append(f"【结论】{conclusion}")
        self.diagnostic_finished.emit("\n".join(lines))

    def show_diagnostic_result(self, text):
        """诊断结果回到 GUI 线程后再弹窗，避免后台线程操作界面。"""
        self.diagnostic_running = False
        self.diag_btn.setEnabled(True)
        self.status_bar.showMessage("连接诊断完成", 5000)
        box = QMessageBox(self)
        box.setWindowTitle("连接诊断结果")
        box.setIcon(QMessageBox.Information)
        box.setText(text)
        box.exec_()

    @staticmethod
    def _ping(host):
        """用系统 ping 判断主机是否可达：2 个包，每个最多等 1 秒。"""
        try:
            creationflags = 0x08000000 if os.name == 'nt' else 0
            result = subprocess.run(
                ["ping", "-n", "2", "-w", "1000", host],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=creationflags
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _probe_port(host, port):
        """尝试建立一次 TCP 连接，返回 (是否成功, 错误说明)。"""
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(4)
        try:
            probe.connect((host, port))
            return True, ""
        except OSError as exc:
            return False, str(exc)
        finally:
            probe.close()

    def _clear_remote_grid(self):
        while self.remote_layout.count():
            item = self.remote_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # 光调 deleteLater() 不够：控件要等事件循环处理 DeferredDelete 才真正
                # 消失，在那之前它会停在原来的位置，和新网格叠在一起。先摘掉父子关系
                # 让它立刻隐藏，再交给 deleteLater 回收。
                widget.setParent(None)
                widget.deleteLater()
        self.remote_tiles.clear()

    def update_conference_members(self, members):
        """成员变化时重建远端视频网格（除自己外最多 MAX_PARTICIPANTS-1 格）。"""
        self.member_order = [str(m.get("id")) for m in members if m.get("id")]
        # ↑ 含自己在内的完整顺序，合成模式要靠它把人对应到格子。
        local_id = self.conference.local_id
        remote_members = [
            m for m in members if m.get("id") != local_id
        ][:MAX_PARTICIPANTS - 1]
        valid_ids = {m.get("id") for m in remote_members}
        if self.audio_link is not None:
            for stale in set(self.remote_frames) - valid_ids:
                self.audio_link.drop_remote(stale)
            # ↑ 人走了就把他的播放缓冲丢掉，否则残留的旧语音还会被混进来
        self.remote_frames = {k: v for k, v in self.remote_frames.items() if k in valid_ids}
        self._clear_remote_grid()
        if not remote_members:
            empty = QLabel("尚无远端成员加入会议")
            empty.setAlignment(Qt.AlignCenter)
            self.remote_layout.addWidget(empty, 0, 0, 1, REMOTE_GRID_COLUMNS)
            return

        for index, member in enumerate(remote_members):
            participant_id = member["id"]
            tile = QWidget()
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(2, 2, 2, 2)
            name_label = QLabel(member.get("name", "参会者"))
            name_label.setAlignment(Qt.AlignCenter)
            image_label = QLabel("等待视频…")
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumSize(145, 82)
            image_label.setStyleSheet("background: #111; color: #ddd;")
            tile_layout.addWidget(image_label)
            tile_layout.addWidget(name_label)
            self.remote_tiles[participant_id] = (image_label, name_label)
            self.remote_layout.addWidget(tile, index // REMOTE_GRID_COLUMNS,
                                         index % REMOTE_GRID_COLUMNS)

    def _set_tile_image(self, participant_id, frame):
        """把一帧 BGR 画面贴到该成员的格子上；转发和合成两种模式共用。"""
        tile = self.remote_tiles.get(participant_id)
        if tile is None:
            return
        image_label, _ = tile
        h, w, channels = frame.shape
        image = QImage(np.ascontiguousarray(frame).data, w, h, channels * w,
                       QImage.Format_BGR888).copy()
        image_label.setPixmap(QPixmap.fromImage(image).scaled(
            image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def display_remote_frame(self, participant_id, name, frame):
        """转发模式：一路远端 BGR 帧已由信号送回主线程，在此安全地显示。"""
        self.remote_frames[participant_id] = frame
        tile = self.remote_tiles.get(participant_id)
        if tile is None:
            return
        tile[1].setText(name)
        self._set_tile_image(participant_id, frame)

    def display_mosaic(self, ver, image):
        """合成模式：从主机发来的一整张多宫格图里，切出每个人对应的那一格。

        切法用的是和主机同一个 mosaic_layout()，只要双方人数一致，格子就一一对应。
        """
        order = self.member_order
        if not order or image is None:
            return
        cols, rows, tile_w, tile_h = mosaic_layout(len(order))
        h, w = image.shape[:2]
        if w < cols * tile_w or h < rows * tile_h:
            return
            # ↑ 人数刚变、双方布局还没对齐时，宁可这一帧不画，也不要切错人。
        for index, participant_id in enumerate(order):
            if participant_id == self.conference.local_id:
                continue
                # ↑ 自己的画面在上面的本地预览里已经有了，这里不再重复占一格。
            row, col = divmod(index, cols)
            y, x = row * tile_h, col * tile_w
            self._set_tile_image(participant_id, image[y:y + tile_h, x:x + tile_w])

    # ---------------- 环境与浏览 ----------------

    def check_environment(self):
        """启动时检查环境，把结果显示在状态栏。"""
        msgs = []
        if torch.cuda.is_available():
            msgs.append(f"CUDA: {torch.cuda.get_device_name(0)}")
        else:
            msgs.append("CUDA 不可用，将使用 CPU")
        if not HAS_SOUNDDEVICE:
            msgs.append("未安装 sounddevice，无法录音与会议语音")
        if find_ffmpeg(self.ffmpeg_edit.text().strip()
                       if hasattr(self, 'ffmpeg_edit') else None) is None:
            msgs.append("未找到 ffmpeg")
        self.status_bar.showMessage(" | ".join(msgs))

    def refresh_audio_devices(self):
        """枚举音频输入/输出设备，并尽量保留用户当前的选择。"""
        inputs, outputs = list_devices()
        self._fill_device_combo(self.mic_combo, inputs, "默认麦克风")
        self._fill_device_combo(self.speaker_combo, outputs, "默认扬声器")

    @staticmethod
    def _fill_device_combo(combo, devices, default_text):
        previous = combo.currentData()
        combo.clear()
        combo.addItem(default_text, None)
        # ↑ userData=None 表示交给 sounddevice 使用系统默认设备
        for index, name in devices:
            combo.addItem(f"{index}: {name}", index)
        if previous is not None:
            position = combo.findData(previous)
            if position >= 0:
                combo.setCurrentIndex(position)
            # ↑ 重新枚举后设备索引可能变号，找不回旧选择就落回默认设备

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重", "", "PyTorch 权重 (*.pth *.pt)"
        )
        # ↑ 返回值是 (路径, 过滤器)。过滤器我们用不到，用 _ 忽略。

        if path:
            self.model_edit.setText(path)

    def browse_background(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择背景图片", "", "图片 (*.jpg *.jpeg *.png *.bmp)"
        )
        if path:
            self.bg_edit.setText(path)

    def browse_save_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        # ↑ 目录选择只有一个返回值
        if path:
            self.save_dir_edit.setText(path)

    def browse_ffmpeg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 ffmpeg 可执行文件", "", "可执行文件 (*.exe);;所有文件 (*)"
        )
        if path:
            self.ffmpeg_edit.setText(path)

    def choose_color(self):
        """选择纯色背景，注意 RGB 与 BGR 的转换。"""
        color = QColorDialog.getColor(QColor(0, 0, 255), self, "选择纯色背景")
        if color.isValid():
            r, g, b = color.red(), color.green(), color.blue()
            # ↑ Qt 用 RGB
            self.bg_color = (b, g, r)
            # ↑ 存 BGR 以匹配 OpenCV
            self.color_label.setText(f"当前颜色: RGB({r},{g},{b})")
            self.color_label.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); color: white;"
            )
            # ↑ 用 CSS 让标签背景变成所选颜色

    def update_threshold(self, value):
        """滑块变化时实时更新线程阈值。"""
        if self.thread and self.thread.isRunning():
            self.thread.threshold = value / 100.0
        # ↑ 直接修改线程属性是线程安全的（Python GIL 保证）

    def update_mirror(self):
        """镜像复选框变化时实时更新线程属性。"""
        if self.thread and self.thread.isRunning():
            self.thread.mirror = self.mirror_check.isChecked()

    # ---------------- 开始/结束视频 ----------------

    def start_video(self):
        """点击开始：校验 → 创建 VideoThread → 启动。"""
        if self.thread and self.thread.isRunning():
            return
            # ↑ 防止重复点击

        # ---- 1. 模型校验 ----
        model_path = self.model_edit.text().strip()
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(self, "警告", "模型文件不存在，请重新选择")
            return
        if not model_path.lower().endswith((".pth", ".pt")):
            QMessageBox.warning(self, "警告", "模型文件扩展名应为 .pth 或 .pt")
            return

        # ---- 2. 背景校验 ----
        if self.bg_image_radio.isChecked():
            bg_path = self.bg_edit.text().strip()
            if not bg_path or not os.path.isfile(bg_path):
                QMessageBox.warning(self, "警告", "背景图片不存在")
                return
            if not bg_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                QMessageBox.warning(self, "警告", "背景图片格式不支持")
                return
            try:
                Image.open(bg_path).verify()
                # ↑ verify() 检查图片文件完整性（不解码像素）
            except Exception as e:
                QMessageBox.warning(self, "警告", f"背景图片无法读取: {e}")
                return
            bg_mode = "image"
            bg_path_val = bg_path
            bg_color_val = None
        else:
            bg_mode = "color"
            bg_path_val = None
            bg_color_val = self.bg_color

        # ---- 3. 保存目录可写性校验 ----
        save_dir = self.save_dir_edit.text().strip() or "outputs"
        try:
            os.makedirs(save_dir, exist_ok=True)
            test_file = os.path.join(save_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            # ↑ 真实写一个文件来测试权限，比 os.access 更可靠
        except Exception as e:
            QMessageBox.warning(self, "警告", f"保存目录不可写: {e}")
            return

        # ---- 4. 设备校验 ----
        device_str = self.device_combo.currentText()
        if device_str == "cuda" and not torch.cuda.is_available():
            QMessageBox.warning(self, "警告", "CUDA 不可用，将使用 CPU")
            device_str = "cpu"

        # ---- 5. 摄像头编号校验 ----
        try:
            camera_id = int(self.camera_combo.currentText())
        except ValueError:
            QMessageBox.warning(self, "警告", "摄像头编号必须为整数")
            return

        # ---- 6. 读取参数 ----
        input_size = int(self.size_combo.currentText())
        threshold = self.threshold_slider.value() / 100.0
        mirror = self.mirror_check.isChecked()
        use_fp16 = self.fp16_check.isChecked()

        # ---- 7. 创建并启动线程 ----
        self.thread = VideoThread(
            camera_id=camera_id,
            model_path=model_path,
            device_str=device_str,
            input_size=input_size,
            threshold=threshold,
            mirror=mirror,
            bg_mode=bg_mode,
            bg_path=bg_path_val,
            bg_color=bg_color_val,
            infer_interval=2,
            use_fp16=use_fp16
        )
        self.thread.frame_ready.connect(self.update_frame)
        self.thread.status_update.connect(self.update_status)
        self.thread.error_signal.connect(self.show_error)
        self.thread.start()

        # ---- 8. 按钮状态切换 ----
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.record_btn.setEnabled(True)

    def stop_video(self):
        """停止录制与视频线程。"""
        self.stop_recording()
        if self.thread and self.thread.isRunning():
            self.thread.stop()
        self.thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(False)

    def update_frame(self, frame):
        """线程发来新帧 → 更新预览，同时写入录制文件。"""
        self.current_frame = frame.copy()
        # ↑ 拷贝一份，避免线程复用同一内存

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        frame_cont = np.ascontiguousarray(frame)
        # ↑ QImage 要求内存连续；ascontiguousarray 保证这一点

        q_img = QImage(frame_cont.data, w, h, bytes_per_line,
                       QImage.Format_BGR888).copy()
        # ↑ 用 BGR888 格式，因为 OpenCV 是 BGR
        #   .copy() 让 QImage 拥有自己的数据副本

        pix = QPixmap.fromImage(q_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        # ↑ 按比例缩放到 label 大小，SmoothTransformation 抗锯齿
        self.video_label.setPixmap(pix)

        # 会议层会自行缩放到 480×270、按发送帧率节流并 JPEG 压缩；因此即使
        # U²-Net 本地预览帧率更高，也不会把网络带宽和 CPU 用在重复帧上。
        self.conference.publish_frame(frame_cont)

        # 若正在录制，写入视频
        if self.recording and self.video_writer is not None:
            try:
                self.video_writer.write(frame_cont)
            except Exception as e:
                print(f"写入视频帧失败: {e}")

    def update_status(self, text):
        self.status_bar.showMessage(text)

    def show_error(self, msg):
        QMessageBox.critical(self, "错误", msg)
        self.stop_video()

    def save_image(self):
        """保存当前合成帧为 JPG。"""
        if self.current_frame is None:
            QMessageBox.warning(self, "警告", "没有可保存的画面")
            return

        save_dir = self.save_dir_edit.text().strip() or "outputs"
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "警告", f"无法创建保存目录: {e}")
            return

        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        # ↑ 时间戳命名避免重复

        path = os.path.join(save_dir, filename)
        try:
            result, n = cv2.imencode(".jpg", self.current_frame,
                                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # ↑ 直接编码到内存，避免 cv2.imwrite 对中文路径的兼容问题
            if not result:
                raise RuntimeError("编码失败")
            with open(path, "wb") as f:
                f.write(n.tobytes())
            self.status_bar.showMessage(f"已保存: {path}", 5000)
            # ↑ 状态栏显示 5 秒
        except Exception as e:
            QMessageBox.warning(self, "警告", f"保存失败: {e}")

    # ---------------- 录制 ----------------

    def start_recording(self):
        """开始录制：创建 VideoWriter，可选启动 AudioRecorder。"""
        if self.recording:
            return
        if self.thread is None or not self.thread.isRunning():
            QMessageBox.warning(self, "警告", "请先开始视频预览")
            return

        save_dir = self.save_dir_edit.text().strip() or "outputs"
        try:
            os.makedirs(save_dir, exist_ok=True)
            test_file = os.path.join(save_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            QMessageBox.warning(self, "警告", f"保存目录不可写: {e}")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.final_path = os.path.join(save_dir, f"record_{ts}.mp4")
        self.record_path = os.path.join(save_dir, f"_temp_video_{ts}.mp4")
        self.audio_path = os.path.join(save_dir, f"_temp_audio_{ts}.wav")
        # ↑ 临时文件用 _temp_ 前缀，合并成功后删除

        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
        else:
            h, w = 360, 640
        # ↑ 用当前帧尺寸作为录制分辨率

        writer = create_video_writer(self.record_path, self.record_fps, (w, h))
        if writer is None:
            QMessageBox.warning(self, "警告", "无法创建视频文件，请检查编码器或路径")
            return

        # ---- 可选：启动录音 ----
        audio_started = False
        if self.record_audio_check.isChecked():
            if not HAS_SOUNDDEVICE:
                QMessageBox.warning(self, "警告",
                                    "未安装 sounddevice，无法录制声音。\n"
                                    "请运行 pip install sounddevice，"
                                    "或取消勾选“录制时包含声音”。")
                writer.release()
                return
            try:
                mic_id = self.mic_combo.currentData()
                self.audio_recorder = AudioRecorder(samplerate=44100, channels=1)
                self.audio_recorder.start(device=mic_id)
                audio_started = True
            except Exception as e:
                QMessageBox.warning(self, "警告", f"无法启动麦克风录音: {e}")
                writer.release()
                self.audio_recorder = None
                return

        # ---- 进入录制状态 ----
        self.video_writer = writer
        self.recording = True
        self.record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(True)
        extra = "（含声音）" if audio_started else "（无声）"
        self.status_bar.showMessage(f"开始录制{extra}: {self.final_path}", 5000)

    def stop_recording(self):
        """停止录制并合并音视频。"""
        if not self.recording:
            return
        self.recording = False

        # ---- 停止视频写入 ----
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None

        # ---- 停止录音 ----
        audio_ok = False
        if self.audio_recorder is not None:
            try:
                audio_ok = self.audio_recorder.stop(self.audio_path)
            except Exception as e:
                print(f"停止录音失败: {e}")
                audio_ok = False
            finally:
                self.audio_recorder = None

        self.record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)

        # ---- 合并或改名 ----
        if audio_ok and os.path.isfile(self.record_path) and os.path.isfile(self.audio_path):
            ffmpeg_path = find_ffmpeg(self.ffmpeg_edit.text().strip())
            if ffmpeg_path is None:
                self.status_bar.showMessage(
                    f"未找到 ffmpeg，已保存无声视频: {self.record_path}", 8000)
                QMessageBox.information(
                    self, "提示",
                    "未找到 ffmpeg，无法合并声音。\n"
                    "已保存无声视频，请安装 ffmpeg 后重试，"
                    "或在程序中手动指定 ffmpeg 路径。"
                )
                return

            self.status_bar.showMessage("正在合并视频与声音...", 0)
            QApplication.processEvents()
            # ↑ 手动触发事件循环，让状态栏立即可见

            ok = self.merge_audio_video(
                ffmpeg_path, self.record_path, self.audio_path, self.final_path
            )
            if ok:
                try:
                    os.remove(self.record_path)
                except Exception:
                    pass
                try:
                    os.remove(self.audio_path)
                except Exception:
                    pass
                self.status_bar.showMessage(f"录制完成: {self.final_path}", 8000)
            else:
                self.status_bar.showMessage(
                    f"合并失败，已保留无声视频: {self.record_path}", 8000)
                QMessageBox.warning(self, "警告", "合并声音失败，请检查 ffmpeg 与文件")
        else:
            # 无音频
            if os.path.isfile(self.record_path):
                try:
                    if os.path.isfile(self.final_path):
                        os.remove(self.final_path)
                    os.rename(self.record_path, self.final_path)
                    self.status_bar.showMessage(
                        f"录制完成（无声）: {self.final_path}", 8000)
                except Exception as e:
                    self.status_bar.showMessage(
                        f"录制完成（无声）: {self.record_path}", 8000)

    def merge_audio_video(self, ffmpeg_path, video_path, audio_path, out_path):
        """用 ffmpeg 合并视频和音频。"""
        try:
            if os.path.isfile(out_path):
                os.remove(out_path)
        except Exception:
            pass

        cmd = [
            ffmpeg_path, '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-shortest',
            out_path
        ]
        # ↑ 参数含义：
        #   -y       : 覆盖输出文件
        #   -i       : 输入文件（可以多个）
        #   -c:v copy: 视频流直接复制，不重编码（快速）
        #   -c:a aac : 音频编码 AAC
        #   -b:a 128k: 音频码率 128kbps
        #   -shortest: 以较短的流为准

        try:
            creationflags = 0
            if os.name == 'nt':
                creationflags = 0x08000000
                # ↑ Windows 下 CREATE_NO_WINDOW = 0x08000000，隐藏控制台

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=creationflags
            )
            if result.returncode != 0:
                print("ffmpeg stderr:")
                print(result.stderr.decode(errors='ignore'))
                return False
            return os.path.isfile(out_path) and os.path.getsize(out_path) > 0
        except Exception as e:
            print(f"ffmpeg 调用失败: {e}")
            return False

    def closeEvent(self, event):
        """关闭窗口时清理资源。"""
        self.leave_conference()
        self.stop_video()
        event.accept()


# ==============================================================================
# 【模块 10：程序入口】
# ==============================================================================

if __name__ == "__main__":
    # ↑ 只有直接运行本文件时才执行；被 import 时不执行
    app = QApplication(sys.argv)
    # ↑ 每个 Qt 程序必须有一个 QApplication 对象
    win = VirtualBackgroundApp()
    win.show()
    sys.exit(app.exec_())
    # ↑ app.exec_() 进入 Qt 事件循环，直到窗口关闭；
    #   sys.exit 用返回码退出
# 仅限作者使用：C:\Users\wanghong\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe
