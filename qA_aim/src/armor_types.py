# -*- coding: utf-8 -*-
"""
装甲板数据结构 —— 移植自 rm_vision / rm_auto_aim
  armor_detector/include/armor_detector/armor.hpp  (Light / Armor)

关键点顺序约定（与题目要求一致）：
  lt = 左灯条上端, lb = 左灯条下端, rb = 右灯条下端, rt = 右灯条上端
"""
from __future__ import annotations

import cv2
import numpy as np

RED = 0
BLUE = 1

COLOR_STR = {RED: "red", BLUE: "blue"}


class Light:
    """灯条。对应 C++ 版 struct Light : public cv::RotatedRect。

    构造过程与 armor.hpp 一致：
      minAreaRect 得到旋转矩形 -> 4 个角点按 y 排序，
      top/bottom 为上/下两条短边的中点，length/width/tilt_angle 由此计算。
    """

    __slots__ = ("center", "top", "bottom", "length", "width",
                 "tilt_angle", "color", "area", "contour")

    def __init__(self, rrect, color=RED, area=0.0, contour=None):
        box = cv2.boxPoints(rrect)                      # 4 角点
        pts = sorted(box, key=lambda p: p[1])           # 按 y 排序
        self.top = (pts[0] + pts[1]) / 2.0
        self.bottom = (pts[2] + pts[3]) / 2.0
        self.length = float(np.linalg.norm(self.top - self.bottom))
        self.width = float(np.linalg.norm(pts[0] - pts[1]))
        # tilt_angle: 灯条与竖直方向的夹角(度), 与 armor.hpp 完全一致
        self.tilt_angle = float(
            np.degrees(np.arctan2(abs(self.top[0] - self.bottom[0]),
                                  abs(self.top[1] - self.bottom[1]))))
        self.center = np.asarray(rrect[0], dtype=np.float64)
        self.color = color
        self.area = area
        self.contour = contour

    @property
    def ratio(self):
        """短边 / 长边"""
        return self.width / max(self.length, 1e-6)

    def __repr__(self):
        return (f"Light(c=({self.center[0]:.1f},{self.center[1]:.1f}) "
                f"l={self.length:.1f} w={self.width:.1f} "
                f"ratio={self.ratio:.2f} ang={self.tilt_angle:.1f} "
                f"color={COLOR_STR[self.color]})")


class Armor:
    """装甲板：由两根灯条配对而成。对应 C++ 版 struct Armor。"""

    __slots__ = ("left_light", "right_light", "center", "type",
                 "number", "confidence")

    def __init__(self, light1: Light, light2: Light):
        # 左灯条 = center.x 较小者 (与 armor.hpp 一致)
        if light1.center[0] < light2.center[0]:
            self.left_light, self.right_light = light1, light2
        else:
            self.left_light, self.right_light = light2, light1
        self.center = (self.left_light.center + self.right_light.center) / 2.0
        self.type = "small"     # "small" / "large" / "invalid"
        self.number = None
        self.confidence = 0.0

    # ---- 题目要求的四个关键点: lt, lb, rb, rt ----
    @property
    def lt(self):
        return self.left_light.top

    @property
    def lb(self):
        return self.left_light.bottom

    @property
    def rb(self):
        return self.right_light.bottom

    @property
    def rt(self):
        return self.right_light.top

    def keypoints(self):
        """返回 [lt, lb, rb, rt] (2,4) 数组"""
        return np.stack([self.lt, self.lb, self.rb, self.rt])

    def pnp_image_points(self):
        """PnP 用的图像点顺序: 左下, 左上, 右上, 右下 (与 pnp_solver.cpp 一致)"""
        return np.stack([self.lb, self.lt, self.rt, self.rb]).astype(np.float64)

    def __repr__(self):
        return (f"Armor(type={self.type} c=({self.center[0]:.1f},{self.center[1]:.1f}) "
                f"L={self.left_light!r} R={self.right_light!r})")
