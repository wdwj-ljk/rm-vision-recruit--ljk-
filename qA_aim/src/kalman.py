# -*- coding: utf-8 -*-
"""
卡尔曼滤波器 —— 平移目标的运动预测 (基本要求 3)

模型: 常速度 (CV) 线性卡尔曼滤波, 状态 [x, vx, y, vy, z, vz],
观测 [x, y, z] (PnP 解算结果), 相机坐标系, 单位 m。
另附常加速度 (CA) 模型用于对比 (发挥部分 Q/R 分析)。

Q、R 物理意义:
  Q = 过程噪声协方差, 由连续白噪声加速度方差 q_acc 驱动;
      表示模型对"目标会机动"的预期, Q 越大滤波越相信新观测、预测越贴运动。
  R = 观测噪声协方差 (PnP 解算噪声), R 越大越平滑但对突变响应越慢。
"""
from __future__ import annotations

import numpy as np


class CVKalman:
    """常速度模型: 状态 [x, vx, y, vy, z, vz]"""

    n_state = 6
    n_obs = 3

    def __init__(self, dt, q_acc=4.0, r_xyz=0.03, p0=None):
        """
        dt     : 滤波周期 (s), 150fps 视频为 1/150
        q_acc  : 连续白噪声加速度方差 (m^2/s^4), 生成过程噪声 Q
        r_xyz  : 单轴观测噪声标准差 (m), R = diag(r_xyz^2) * I3
        p0     : 初始状态协方差 (可传入标量对角)
        """
        self.dt = dt
        self.q_acc = q_acc
        self.r_xyz = r_xyz
        self.F = self._build_F(dt)
        self.H = np.zeros((self.n_obs, self.n_state))
        self.H[0, 0] = self.H[1, 2] = self.H[2, 4] = 1.0
        self.Q = self._build_Q(dt, q_acc)
        self.R = np.eye(self.n_obs) * r_xyz ** 2
        if p0 is None:
            self.P = np.eye(self.n_state) * 1.0
        else:
            self.P = np.eye(self.n_state) * p0
        self.x = None

    @staticmethod
    def _build_F(dt):
        F = np.eye(6)
        for i in (0, 2, 4):
            F[i, i + 1] = dt
        return F

    @staticmethod
    def _build_Q(dt, q_acc):
        # 连续白噪声加速度 -> 离散过程噪声 (CV)
        q = np.zeros((6, 6))
        for i in (0, 2, 4):
            q[i:i + 2, i:i + 2] = np.array(
                [[dt ** 4 / 4.0, dt ** 3 / 2.0],
                 [dt ** 3 / 2.0, dt ** 2]]) * q_acc
        return q

    @property
    def initialized(self):
        return self.x is not None

    def init(self, pos, vel=None):
        self.x = np.zeros(6)
        self.x[0], self.x[2], self.x[4] = pos
        if vel is not None:
            self.x[1], self.x[3], self.x[5] = vel
        self.P = np.eye(6) * 1.0
        return self.x

    def predict(self, dt=None):
        """一步预测并推进时间"""
        if self.x is None:
            return None
        if dt is not None and abs(dt - self.dt) > 1e-12:
            F = self._build_F(dt)
            Q = self._build_Q(dt, self.q_acc)
        else:
            F, Q = self.F, self.Q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x

    def update(self, z):
        """用观测 z=[x,y,z] 更新"""
        if self.x is None:
            return None
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.n_state) - K @ self.H) @ self.P
        return self.x

    def predict_ahead(self, total_dt, n_steps=None):
        """预测 total_dt 秒后的状态 (不改变滤波器内部状态)。
        返回位置 [x,y,z]。将 total_dt 拆成整数步 + 余数步。"""
        if self.x is None:
            return None
        x = self.x.copy()
        F = self.F
        steps = int(round(total_dt / self.dt))
        rest = total_dt - steps * self.dt
        for _ in range(steps):
            x = F @ x
        if abs(rest) > 1e-9:
            x = self._build_F(rest) @ x
        return np.array([x[0], x[2], x[4]])

    def pos(self):
        return np.array([self.x[0], self.x[2], self.x[4]])

    def copy(self):
        kf = CVKalman(self.dt, self.q_acc, self.r_xyz)
        kf.x = self.x.copy() if self.x is not None else None
        kf.P = self.P.copy()
        return kf


class CAKalman:
    """常加速度模型: 状态 [x, vx, ax, y, vy, ay, z, vz, az], 用于对比实验"""

    n_state = 9
    n_obs = 3

    def __init__(self, dt, q_jerk=50.0, r_xyz=0.03):
        self.dt = dt
        self.q_jerk = q_jerk
        self.r_xyz = r_xyz
        self.F = self._build_F(dt)
        self.H = np.zeros((3, 9))
        self.H[0, 0] = self.H[1, 3] = self.H[2, 6] = 1.0
        self.Q = self._build_Q(dt, q_jerk)
        self.R = np.eye(3) * r_xyz ** 2
        self.P = np.eye(9) * 1.0
        self.x = None

    @staticmethod
    def _build_F(dt):
        F = np.eye(9)
        for i in (0, 3, 6):
            F[i, i + 1] = dt
            F[i, i + 2] = dt ** 2 / 2.0
            F[i + 1, i + 2] = dt
        return F

    @staticmethod
    def _build_Q(dt, q_jerk):
        q = np.zeros((9, 9))
        for i in (0, 3, 6):
            q[i:i + 3, i:i + 3] = np.array(
                [[dt ** 6 / 36.0, dt ** 5 / 12.0, dt ** 4 / 6.0],
                 [dt ** 5 / 12.0, dt ** 4 / 4.0, dt ** 3 / 2.0],
                 [dt ** 4 / 6.0, dt ** 3 / 2.0, dt ** 2]]) * q_jerk
        return q

    @property
    def initialized(self):
        return self.x is not None

    def init(self, pos):
        self.x = np.zeros(9)
        self.x[0], self.x[3], self.x[6] = pos
        self.P = np.eye(9) * 1.0
        return self.x

    def predict(self, dt=None):
        if self.x is None:
            return None
        F = self._build_F(dt) if dt is not None else self.F
        Q = self._build_Q(dt, self.q_jerk) if dt is not None else self.Q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x

    def update(self, z):
        if self.x is None:
            return None
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ self.H) @ self.P
        return self.x

    def predict_ahead(self, total_dt):
        if self.x is None:
            return None
        x = self.x.copy()
        F = self.F
        steps = int(round(total_dt / self.dt))
        rest = total_dt - steps * self.dt
        for _ in range(steps):
            x = F @ x
        if abs(rest) > 1e-9:
            x = self._build_F(rest) @ x
        return np.array([x[0], x[3], x[6]])

    def pos(self):
        return np.array([self.x[0], self.x[3], self.x[6]])

    def copy(self):
        kf = CAKalman(self.dt, self.q_jerk, self.r_xyz)
        kf.x = self.x.copy() if self.x is not None else None
        kf.P = self.P.copy()
        return kf
