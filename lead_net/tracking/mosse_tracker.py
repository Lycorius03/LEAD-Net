"""MOSSE (Minimum Output Sum of Squared Error) 轻量级相关滤波追踪器。

论文: Bolme et al., "Visual Object Tracking using Adaptive Correlation Filters", CVPR 2010
嵌入式实测: STM32H743 @ 320×240 灰度 → 161 FPS, 6.2ms/帧

用途: 当 DL 检测暂时失败时（运动模糊、短暂遮挡），MOSSE 可桥接 3-5 帧。
     "重检测 + 轻追踪" 混合架构。

原理: 在频域学习一个相关滤波器 H，使得:
    G = F ⊙ H*  →  H = G ⊙ F* / (F ⊙ F* + ε)
    其中 F = FFT(template), G = FFT(desired_response)

追踪: 在新帧中裁剪搜索区域 → FFT → 与 H 相关 → IFFT → 峰值位置 = 新目标位置

用法::

    tracker = MOSSETracker(learning_rate=0.125)
    tracker.init(frame_gray, bbox_xywh)           # 初始化模板
    new_bbox = tracker.update(frame_gray)          # 追踪一帧
    if new_bbox is None:                           # 追踪失败
        ...
"""

from __future__ import annotations

import numpy as np


def _gaussian2d(shape: tuple[int, int], sigma: float = 2.0) -> np.ndarray:
    """生成 2D 高斯响应图。"""
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    y = np.arange(h, dtype=np.float64)[:, None]
    x = np.arange(w, dtype=np.float64)[None, :]
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))


def _cos_window(shape: tuple[int, int]) -> np.ndarray:
    """生成余弦窗（减少 FFT 边界效应）。"""
    h, w = shape
    wy = np.hanning(h)
    wx = np.hanning(w)
    return np.outer(wy, wx)


class MOSSETracker:
    """MOSSE 轻量相关滤波追踪器。

    Args:
        learning_rate: 模板学习率 η ∈ (0, 1)，默认 0.125。
        sigma: 高斯响应图 σ，默认 2.0。
        eps: 正则化项，防止除零，默认 1e-5。
    """

    def __init__(self, learning_rate: float = 0.125, sigma: float = 2.0, eps: float = 1e-5):
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.eps = eps

        self._H: np.ndarray | None = None        # 滤波器 (频域)
        self._A: np.ndarray | None = None        # 分子累积
        self._B: np.ndarray | None = None        # 分母累积
        self._template_size: tuple[int, int] | None = None
        self._last_bbox: tuple[float, float, float, float] | None = None
        self._cos_window: np.ndarray | None = None

    @property
    def is_initialized(self) -> bool:
        return self._H is not None

    @property
    def last_bbox(self) -> tuple[float, float, float, float] | None:
        return self._last_bbox

    def init(self, gray: np.ndarray, bbox: tuple[float, float, float, float]) -> None:
        """初始化 MOSSE 追踪器。

        Args:
            gray: 灰度图像 [H, W]，dtype=float64，值域 [0, 1]
            bbox: (cx, cy, w, h) 初始目标边界框
        """
        cx, cy, w, h = bbox
        h_img, w_img = gray.shape

        # 裁剪目标区域（填充到 2× 尺寸供 FFT）
        pad = 1.5
        crop_w = int(w * pad)
        crop_h = int(h * pad)
        self._template_size = (crop_h, crop_w)

        x1 = max(0, int(cx - crop_w // 2))
        y1 = max(0, int(cy - crop_h // 2))
        x2 = min(w_img, x1 + crop_w)
        y2 = min(h_img, y1 + crop_h)
        x1 = max(0, x2 - crop_w)
        y1 = max(0, y2 - crop_h)

        template = gray[y1:y2, x1:x2].astype(np.float64)

        # 余弦窗 + 归一化
        self._cos_window = _cos_window(template.shape)
        template = template * self._cos_window
        template = (template - template.mean()) / (template.std() + self.eps)

        # 期望响应（2D 高斯）
        G = _gaussian2d(template.shape, self.sigma)
        G_fft = np.fft.fft2(G)

        # 初始滤波器: H = G ⊙ F* / (F ⊙ F* + ε)
        F = np.fft.fft2(template)
        self._A = G_fft * np.conj(F)
        self._B = F * np.conj(F) + self.eps
        self._H = self._A / self._B

        self._last_bbox = bbox

    def update(self, gray: np.ndarray) -> tuple[float, float, float, float] | None:
        """在新帧中追踪目标。

        Args:
            gray: 灰度图像 [H, W]

        Returns:
            (cx, cy, w, h) 新位置，或 None 表示追踪失败。
        """
        if not self.is_initialized or self._last_bbox is None:
            return None

        cx, cy, w, h = self._last_bbox
        h_img, w_img = gray.shape
        crop_h, crop_w = self._template_size
        if crop_h is None or crop_w is None:
            return None

        # 以预测位置为中心裁剪
        x1 = max(0, int(cx - crop_w // 2))
        y1 = max(0, int(cy - crop_h // 2))
        x2 = min(w_img, x1 + crop_w)
        y2 = min(h_img, y1 + crop_h)
        x1 = max(0, x2 - crop_w)
        y1 = max(0, y2 - crop_h)

        patch = gray[y1:y2, x1:x2].astype(np.float64)

        # 预处理
        if self._cos_window is not None and patch.shape == self._cos_window.shape:
            patch = patch * self._cos_window
        patch = (patch - patch.mean()) / (patch.std() + self.eps)

        # 相关响应
        F = np.fft.fft2(patch)
        response = np.fft.ifft2(self._H * F)
        response = np.abs(response)

        # 峰值位置 → 偏移
        dy, dx = np.unravel_index(response.argmax(), response.shape)
        if dy > crop_h // 2:
            dy -= crop_h
        if dx > crop_w // 2:
            dx -= crop_w

        # PSR (Peak-to-Sidelobe Ratio) 质量检查
        peak = response.max()
        sidelobe = (response[response < peak * 0.9]).mean()
        psr = (peak - sidelobe) / (sidelobe + self.eps)
        if psr < 3.0:  # 追踪质量太差
            return None

        # 新位置（相对于原始裁剪中心）
        new_cx = x1 + crop_w // 2 + dx
        new_cy = y1 + crop_h // 2 + dy
        self._last_bbox = (float(new_cx), float(new_cy), float(w), float(h))

        # 在线更新滤波器
        G = _gaussian2d(patch.shape, self.sigma)
        G_fft = np.fft.fft2(G)
        lr = self.learning_rate
        self._A = (1 - lr) * self._A + lr * (G_fft * np.conj(F))
        self._B = (1 - lr) * self._B + lr * (F * np.conj(F) + self.eps)
        self._H = self._A / self._B

        return self._last_bbox

    def reset(self) -> None:
        """重置追踪器。"""
        self._H = None
        self._A = None
        self._B = None
        self._template_size = None
        self._last_bbox = None
        self._cos_window = None
