# -*- coding: utf-8 -*-
"""
装甲板检测器 —— 移植自 rm_vision / rm_auto_aim 的 armor_detector 包
  armor_detector/src/detector.cpp     -> Detector (preprocessImage / findLights / matchLights)
  armor_detector/src/detector_node.cpp-> 默认参数

相对原仓库的改动（视频模式所需）：
  1. 阈值模式: 原仓库依赖低曝光相机, 直接对灰度图二值化;
     这里保留 'gray' 模式, 并新增 'color' 模式 (通道差分 R-max(G,B) / B-max(G,R)),
     对普通实拍视频更鲁棒。
  2. 灯条分段合并: 原仓库曝光调好后灯条连续; 实拍视频中反光/遮挡会把
     一条灯条断成几段, 因此在配对前先把共线且贴近的灯条段合并。
  3. 数字分类器(MLP)未移植——本题视频中的测试台不含数字, 改用
     几何约束 + 时间连续性筛选。
"""
from __future__ import annotations

import cv2
import numpy as np

from .armor_types import Armor, Light, RED, BLUE


class LightParams:
    """灯条筛选参数 (detector_node.cpp 默认值)"""

    def __init__(self):
        self.min_ratio = 0.1      # 短边/长边 下界
        self.max_ratio = 0.4      # 短边/长边 上界
        self.max_angle = 40.0     # 灯条倾角上界 (度)
        self.min_area = 20.0      # 轮廓最小面积 (px^2), 视频模式新增


class ArmorParams:
    """灯条配对参数 (detector_node.cpp 默认值)"""

    def __init__(self):
        self.min_light_ratio = 0.7           # 两灯条长度比下界
        self.min_small_center_distance = 0.8  # 小装甲板中心距/灯长
        self.max_small_center_distance = 3.2
        self.min_large_center_distance = 3.2  # 大装甲板中心距/灯长
        self.max_large_center_distance = 5.5
        self.max_angle = 35.0                 # 两灯条中心连线倾角上界 (度)


class Detector:
    """Detector —— detector.cpp 的 Python 移植"""

    def __init__(self, binary_thres=160, detect_color=RED,
                 thres_mode="color", min_brightness=60,
                 light_params=None, armor_params=None,
                 merge_segments=True,
                 merge_angle_tol=15.0, merge_gap=0.8, merge_lateral=0.6):
        self.binary_thres = binary_thres
        self.detect_color = detect_color
        self.thres_mode = thres_mode
        self.min_brightness = min_brightness
        self.l = light_params or LightParams()
        self.a = armor_params or ArmorParams()
        self.merge_segments = merge_segments
        self.merge_angle_tol = merge_angle_tol
        self.merge_gap = merge_gap
        self.merge_lateral = merge_lateral

        self.binary_img = None
        self.lights = []
        self.armors = []

    # ------------------------------------------------------------------
    def detect(self, img_bgr):
        """主流程 (对应 detector.cpp::detect)"""
        self.binary_img = self.preprocess_image(img_bgr)
        self.lights = self.find_lights(img_bgr, self.binary_img)
        if self.merge_segments:
            self.lights = self.merge_light_segments(self.lights)
        self.armors = self.match_lights(self.lights)
        return self.armors

    def preprocess_image(self, img_bgr):
        """二值化 (对应 detector.cpp::preprocessImage, 增加 color 模式)"""
        if self.thres_mode == "gray":
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, self.binary_thres, 255,
                                      cv2.THRESH_BINARY)
        else:  # color: 红色 R - max(G,B), 蓝色 B - max(G,R)
            b, g, r = (img_bgr[..., 0].astype(np.int16),
                       img_bgr[..., 1].astype(np.int16),
                       img_bgr[..., 2].astype(np.int16))
            dom = r - np.maximum(g, b) if self.detect_color == RED \
                else b - np.maximum(g, r)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            binary = np.where((dom > self.binary_thres) &
                              (gray > self.min_brightness), 255, 0).astype(np.uint8)
        return binary

    def find_lights(self, img_bgr, binary):
        """找灯条 (对应 detector.cpp::findLights)"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        lights = []
        for contour in contours:
            # 原仓库要求 contour.size() >= 5; 视频里远距离灯条只有 4 个角点,
            # 故放宽为 >= 4
            if len(contour) < 4:
                continue
            rrect = cv2.minAreaRect(contour)
            light = Light(rrect, color=RED, area=cv2.contourArea(contour),
                          contour=contour)
            if not self._is_light(light):
                continue
            # 灯条颜色判定 (detector.cpp: 轮廓内红色像素和 > 蓝色像素和 -> RED)
            x, y, w, h = cv2.boundingRect(contour)
            if 0 <= x and 0 <= y and x + w <= img_bgr.shape[1] \
                    and y + h <= img_bgr.shape[0] and w > 0 and h > 0:
                roi = img_bgr[y:y + h, x:x + w]
                mask = np.zeros((h, w), np.uint8)
                cv2.drawContours(mask, [contour - (x, y)], -1, 255, -1)
                sum_b = int(roi[..., 0][mask > 0].sum())
                sum_r = int(roi[..., 2][mask > 0].sum())
                light.color = RED if sum_r > sum_b else BLUE
                lights.append(light)
        return lights

    def _is_light(self, light):
        """灯条判定 (对应 detector.cpp::isLight)"""
        ratio_ok = self.l.min_ratio < light.ratio < self.l.max_ratio
        angle_ok = light.tilt_angle < self.l.max_angle
        area_ok = light.area >= self.l.min_area
        return ratio_ok and angle_ok and area_ok

    # ------------------------------------------------------------------
    # 视频模式新增: 灯条分段合并
    # ------------------------------------------------------------------
    @staticmethod
    def _collinear_close(m, other, angle_tol, gap_tol, lateral_tol):
        """判断 other 是否为 m 的断裂段: 共线、横向偏移小、轴向间隙小"""
        if m.length < 1e-6 or other.length < 1e-6:
            return False
        u_m = (m.bottom - m.top) / m.length
        u_o = (other.bottom - other.top) / other.length
        cosang = abs(float(np.dot(u_m, u_o)))
        if np.degrees(np.arccos(min(1.0, cosang))) > angle_tol:
            return False
        # 横向偏移: other 中心到 m 轴线的距离
        lateral = abs(float(u_m[0] * (other.center[1] - m.top[1]) -
                            u_m[1] * (other.center[0] - m.top[0])))
        if lateral > lateral_tol * (m.width + other.width) / 2.0 + 2.0:
            return False
        # 轴向间隙: 投影距离 - 半长和
        proj = float(np.dot(u_m, other.center - m.center))
        gap = abs(proj) - (m.length + other.length) / 2.0
        return gap < gap_tol * (m.length + other.length) / 2.0

    def merge_light_segments(self, lights):
        """把共线断裂的灯条段合并为一条 (视频模式新增)"""
        used = [False] * len(lights)
        merged = []
        for i, li in enumerate(lights):
            if used[i]:
                continue
            group = [li]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j, lj in enumerate(lights):
                    if used[j] or lj.color != li.color:
                        continue
                    if any(self._collinear_close(m, lj, self.merge_angle_tol,
                                                 self.merge_gap,
                                                 self.merge_lateral)
                           for m in group):
                        group.append(lj)
                        used[j] = True
                        changed = True
            if len(group) == 1:
                merged.append(li)
            else:
                pts = np.vstack([l.contour for l in group])
                rrect = cv2.minAreaRect(pts)
                merged.append(Light(rrect, color=li.color,
                                    area=sum(l.area for l in group),
                                    contour=pts))
        return merged

    # ------------------------------------------------------------------
    # 灯条配对 (对应 detector.cpp::matchLights / containLight / isArmor)
    # ------------------------------------------------------------------
    def match_lights(self, lights):
        armors = []
        for i in range(len(lights)):
            for j in range(i + 1, len(lights)):
                light1, light2 = lights[i], lights[j]
                if light1.color != self.detect_color or \
                        light2.color != self.detect_color:
                    continue
                if self._contain_light(light1, light2, lights):
                    continue
                atype = self._is_armor(light1, light2)
                if atype is not None:
                    armor = Armor(light1, light2)
                    armor.type = atype
                    armors.append(armor)
        return armors

    @staticmethod
    def _contain_light(light1, light2, lights):
        """两灯条包围盒内是否有第三条灯条 (detector.cpp::containLight)"""
        points = [light1.top, light1.bottom, light2.top, light2.bottom]
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)
        for tl in lights:
            if np.array_equal(tl.center, light1.center) or \
                    np.array_equal(tl.center, light2.center):
                continue
            for p in (tl.top, tl.bottom, tl.center):
                if x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
                    return True
        return False

    def _is_armor(self, light1, light2):
        """配对判定 (detector.cpp::isArmor), 返回 'small'/'large'/None"""
        # 两灯条长度比 (短/长)
        length_ratio = (min(light1.length, light2.length) /
                        max(light1.length, light2.length))
        light_ratio_ok = length_ratio > self.a.min_light_ratio

        # 中心距 / 平均灯长
        avg_len = (light1.length + light2.length) / 2.0
        center_distance = (float(np.linalg.norm(light1.center - light2.center)) /
                           avg_len)
        center_distance_ok = (
            self.a.min_small_center_distance <= center_distance <
            self.a.max_small_center_distance) or (
            self.a.min_large_center_distance <= center_distance <
            self.a.max_large_center_distance)

        # 中心连线倾角
        diff = light1.center - light2.center
        angle = abs(np.degrees(np.arctan2(diff[1], diff[0])))
        angle_ok = angle < self.a.max_angle

        if not (light_ratio_ok and center_distance_ok and angle_ok):
            return None
        return "large" if center_distance >= self.a.min_large_center_distance \
            else "small"

    # ------------------------------------------------------------------
    def draw_results(self, img_bgr, draw_lights=True, pose_texts=None):
        """可视化 (对应 detector.cpp::drawResults, 增加关键点序号标注)"""
        img = img_bgr.copy()
        if draw_lights:
            for light in self.lights:
                if light.color != self.detect_color:
                    continue
                c = (0, 0, 255) if light.color == RED else (255, 0, 0)
                cv2.circle(img, tuple(light.top.astype(int)), 3, (255, 255, 255), 1)
                cv2.circle(img, tuple(light.bottom.astype(int)), 3, (255, 255, 255), 1)
                cv2.line(img, tuple(light.top.astype(int)),
                         tuple(light.bottom.astype(int)), c, 1)
        for armor in self.armors:
            pts = armor.keypoints().astype(int)  # lt, lb, rb, rt
            cv2.line(img, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2)
            cv2.line(img, tuple(pts[1]), tuple(pts[2]), (0, 255, 0), 2)
            cv2.line(img, tuple(pts[2]), tuple(pts[3]), (0, 255, 0), 2)
            cv2.line(img, tuple(pts[3]), tuple(pts[0]), (0, 255, 0), 2)
            labels = ["lt", "lb", "rb", "rt"]
            for k, p in enumerate(pts):
                cv2.circle(img, tuple(p), 4, (0, 255, 255), -1)
                cv2.putText(img, labels[k],
                            tuple(p + np.array([4, -4])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if pose_texts:
            for i, txt in enumerate(pose_texts):
                cv2.putText(img, txt, (10, 30 + 24 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return img
