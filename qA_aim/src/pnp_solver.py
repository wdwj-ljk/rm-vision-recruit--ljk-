# -*- coding: utf-8 -*-
"""
PnP 位姿解算 —— 移植自 rm_vision / rm_auto_aim
  armor_detector/src/pnp_solver.cpp

装甲板物理尺寸 (mm):
  小装甲板: 灯条长 55, 两灯条中心距 135
  大装甲板: 灯条长 55, 两灯条中心距 230
坐标系: OpenCV 相机坐标系 x 向右 / y 向下 / z 向前, 单位 m。
"""
from __future__ import annotations

import cv2
import numpy as np

SMALL_ARMOR_WIDTH = 135.0   # mm
SMALL_ARMOR_HEIGHT = 55.0   # mm
LARGE_ARMOR_WIDTH = 230.0   # mm
LARGE_ARMOR_HEIGHT = 55.0   # mm


class PnPSolver:
    """PnPSolver —— pnp_solver.cpp 的 Python 移植"""

    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)

        # 模型坐标: x 前, y 左, z 上; 从左下角顺时针 (与 pnp_solver.cpp 一致)
        def plate_points(width_mm, height_mm):
            hy = width_mm / 2.0 / 1000.0
            hz = height_mm / 2.0 / 1000.0
            return np.array([
                [0.0, +hy, -hz],   # 左下 (left bottom)
                [0.0, +hy, +hz],   # 左上 (left top)
                [0.0, -hy, +hz],   # 右上 (right top)
                [0.0, -hy, -hz],   # 右下 (right bottom)
            ], dtype=np.float64)

        self.small_armor_points = plate_points(SMALL_ARMOR_WIDTH,
                                               SMALL_ARMOR_HEIGHT)
        self.large_armor_points = plate_points(LARGE_ARMOR_WIDTH,
                                               LARGE_ARMOR_HEIGHT)

    def solve(self, armor):
        """返回 (tvec, yaw) 或 (None, None)。
        tvec: 装甲板中心在相机坐标系的位置 (m)
        yaw:  装甲板朝向角 (rad), 正对相机为 0, 逆时针(俯视)为正
        """
        image_points = armor.pnp_image_points()  # lb, lt, rt, rb
        object_points = (self.small_armor_points if armor.type == "small"
                         else self.large_armor_points)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points, image_points, self.camera_matrix,
                self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points, image_points, self.camera_matrix,
                    self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None, None
        except cv2.error:
            return None, None

        tvec = tvec.reshape(3)
        if tvec[2] <= 0:      # 解在相机后方, 非法
            return None, None
        R, _ = cv2.Rodrigues(rvec)
        # 装甲板法向量 n = R * [0,0,1]; yaw 为法向量在 x-z 平面的朝向角
        yaw = float(np.arctan2(R[0, 2], R[2, 2]))
        return tvec, yaw

    def calculate_distance_to_center(self, image_point):
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        return float(np.linalg.norm(np.asarray(image_point, dtype=np.float64) -
                                    np.array([cx, cy])))


def load_camera_info(path):
    """读取考核提供的 camera_info.json"""
    import json
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)
    K = np.array(info["camera_matrix"], dtype=np.float64)
    D = np.array(info["distortion_coefficients"], dtype=np.float64)
    return K, D
