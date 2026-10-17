# -*- coding: utf-8 -*-
"""
小陀螺目标估计 —— 发挥部分 (2)

原理:
  测试台原地旋转, 各装甲板绕车体中心做圆周运动。相机只能看到正对
  相机的板, 因此观测到的装甲板朝向角 yaw 呈锯齿波: 在可见区间内
  以角速度 omega 线性扫过约 ±45°~±60°, 换板时跳变约 90°。
  于是:
    1. 按 yaw 连续性把观测分为若干"扫掠段", 每段内 yaw 对时间做
       线性拟合, 斜率即该段的角速度;
    2. 对各段斜率取加权中位数 (权重=段长) 得全局角速度 omega;
    3. 车体中心 = 观测位置的中位数 (板绕中心对称分布);
    4. 旋转半径由观测相对中心距离的分布估计。
  该方法对 PnP 位置噪声、板切换、误配对天然鲁棒。
"""
from __future__ import annotations

import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def segment_slopes(frames, yaw, max_gap=3, seg_thresh=0.4, min_n=3):
    """把 (frames, yaw) 序列分为扫掠段并返回每段 (斜率, 段长, 段中点时间)"""
    frames = np.asarray(frames, dtype=np.int64)
    yaw = np.asarray(yaw, dtype=np.float64)
    order = np.argsort(frames)
    frames, yaw = frames[order], yaw[order]
    n = len(frames)
    if n == 0:
        return np.empty((0, 3))
    seg = np.zeros(n, dtype=np.int64)
    sid = 0
    for i in range(1, n):
        dtf = frames[i] - frames[i - 1]
        if dtf <= max_gap and abs(wrap(yaw[i] - yaw[i - 1])) < seg_thresh:
            seg[i] = sid
        else:
            sid += 1
            seg[i] = sid
    out = []
    for s in np.unique(seg):
        m = seg == s
        if m.sum() < min_n:
            continue
        tt = frames[m].astype(np.float64) / 150.0
        yy = yaw[m]
        A = np.column_stack([np.ones_like(tt), tt])
        sol, *_ = np.linalg.lstsq(A, yy, rcond=None)
        out.append((sol[1], float(m.sum()), float(tt.mean())))
    return np.array(out)


def assign_segments(frames, yaw, max_gap=3, seg_thresh=0.4):
    """返回与输入 frames 对齐的段编号数组 (未排序保持原序)"""
    frames = np.asarray(frames, dtype=np.int64)
    yaw = np.asarray(yaw, dtype=np.float64)
    order = np.argsort(frames)
    frames_s, yaw_s = frames[order], yaw[order]
    seg_s = np.zeros(len(frames_s), dtype=np.int64)
    sid = 0
    for i in range(1, len(frames_s)):
        dtf = frames_s[i] - frames_s[i - 1]
        if dtf <= max_gap and abs(wrap(yaw_s[i] - yaw_s[i - 1])) < seg_thresh:
            seg_s[i] = sid
        else:
            sid += 1
            seg_s[i] = sid
    seg = np.empty(len(frames), dtype=np.int64)
    seg[order] = seg_s
    return seg


def weighted_median(values, weights=None):
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    idx = np.argsort(values)
    values, weights = values[idx], weights[idx]
    cw = np.cumsum(weights)
    return float(values[np.searchsorted(cw, cw[-1] / 2.0)])


def estimate_gyro(frames, yaw, obs, strong_thresh=5.0, min_strong=3):
    """输入:
      frames: (N,) 帧号 (150fps)
      yaw:    (N,) 装甲板朝向角 (rad)
      obs:    (N,3) PnP 位置 [x,y,z] (m)
    输出 dict:
      cx, cy, cz    车体中心 (m)
      r_est         旋转半径估计 (m)
      omega_global  全局角速度 (rad/s)
      segs          每段 (斜率, 段长, 段中点时间)
      omega_local   逐帧角速度 (与输入 frames 对齐, 无观测段用全局值)
    """
    result = {}
    result["cx"] = float(np.median(obs[:, 0]))
    result["cy"] = float(np.median(obs[:, 1]))
    result["cz"] = float(np.median(obs[:, 2]))
    d = np.sqrt((obs[:, 0] - result["cx"]) ** 2 +
                (obs[:, 2] - result["cz"]) ** 2)
    result["r_est"] = float(np.median(d))

    segs = segment_slopes(frames, yaw, min_n=3)
    result["segs"] = segs
    if len(segs) == 0:
        result["omega_global"] = None
        result["omega_local"] = np.full(len(frames), np.nan)
        return result

    slopes = segs[:, 0]
    # 信息段: |omega| > strong_thresh 的段 (排除噪声/混板段), 不足则用全部
    strong = np.abs(slopes) > strong_thresh
    pool = segs[strong] if strong.sum() >= min_strong else segs
    result["omega_global"] = weighted_median(pool[:, 0], pool[:, 1])
    result["n_segments"] = len(segs)
    result["n_strong"] = int(strong.sum())

    # 逐帧角速度: 各段斜率, 短段(<5 帧)不可信则用全局值
    seg_ids = assign_segments(frames, yaw)
    segs5 = segment_slopes(frames, yaw, min_n=5)
    good_ids = set()
    slopes_by_id = {}
    # 重新计算含段编号的斜率表
    segs_full = []
    for s in np.unique(seg_ids):
        m = seg_ids == s
        tt = frames[m].astype(np.float64) / 150.0
        yy = yaw[m]
        A = np.column_stack([np.ones_like(tt), tt])
        sol, *_ = np.linalg.lstsq(A, yy, rcond=None)
        segs_full.append((s, sol[1], int(m.sum())))
    segs_full = np.array(segs_full, dtype=object)
    for s, sl, cnt in segs_full:
        if cnt >= 5:
            good_ids.add(s)
            slopes_by_id[s] = sl
    omega_local = np.full(len(frames), np.nan)
    for i in range(len(frames)):
        s = seg_ids[i]
        if s in good_ids:
            omega_local[i] = slopes_by_id[s]
    # 短段/间隙用全局值填充, 再中值平滑
    fill = np.isnan(omega_local)
    omega_local[fill] = result["omega_global"]
    k = 5
    smoothed = omega_local.copy()
    for i in range(len(omega_local)):
        lo, hi = max(0, i - k), min(len(omega_local), i + k + 1)
        smoothed[i] = np.median(omega_local[lo:hi])
    result["omega_local"] = smoothed
    return result
