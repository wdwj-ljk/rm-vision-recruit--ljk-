# -*- coding: utf-8 -*-
"""
平移目标运动预测 —— 基本要求 (2)(3)

流程:
  1. 读取 detections_远距离平移.csv (PnP 位姿序列)
  2. 位姿观测清洗: z 与灯条长度反算深度一致性检验 (剔除错误配对/坏位姿)
  3. 卡尔曼滤波 (常速度 CV / 常加速度 CA 可选), 数据关联取与预测最近者
  4. 对每一帧预测 0.25 s (37.5 帧) 后的位置
  5. 输出:
     - pred_远距离平移.csv  (t, x, y, z 为 t+0.25s 的预测)
     - 位置随时间变化曲线、x-z 俯视轨迹
     - 观测/滤波/预测对比曲线、预测值 vs 0.25s 后观测值对比曲线与误差统计
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.kalman import CVKalman, CAKalman                 # noqa: E402
from src.video_utils import (error_stats, save_fig,       # noqa: E402
                             setup_chinese_font, write_pred_csv)

DATA = Path(r"C:\Users\Lenovo\Desktop\给新生 2\data\armor")
FPS = 150.0
PRED_HORIZON = 0.25          # s
FY = 2322.7                  # 由 camera_info.json


def load_detections(csv_path):
    """读取 detections CSV -> (t, x, y, z, bar_len_px, w_px, type) 数组"""
    import csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r["x"] == "" or r["z"] == "":
                continue
            frame = int(float(r["frame"]))
            lt = np.array([float(r["lt_x"]), float(r["lt_y"])])
            lb = np.array([float(r["lb_x"]), float(r["lb_y"])])
            rb = np.array([float(r["rb_x"]), float(r["rb_y"])])
            bar_len = float(np.linalg.norm(lt - lb))
            w_px = float(np.linalg.norm(lb - rb))
            rows.append((frame, float(r["x"]), float(r["y"]), float(r["z"]),
                         float(r["yaw"]), bar_len, w_px))
    return rows


def sanity_filter(rows, z_tol=0.22):
    """深度一致性检验: |z_pose - z_barlen|/z_pose < z_tol"""
    out = []
    for r in rows:
        frame, x, y, z, yaw, bar_len, w_px = r
        z_bar = FY * 0.055 / max(bar_len, 1e-6)
        if z > 0.5 and abs(z - z_bar) / z < z_tol and abs(yaw) < 1.5:
            out.append(r)
    return out


def run_tracking(obs_list, total_frames, model="cv",
                 q_acc=4.0, r_xyz=0.03, q_jerk=50.0, gate=0.4,
                 yaw_gate=0.45, max_miss=20, vmax=3.0):
    """逐帧 KF 滤波 + 0.25s 预测。
    obs_list: (frame, x, y, z, yaw, bar_len, w_px) 过滤后的观测
    数据关联: 位置最近 + yaw 连续 (避免在轨道端点错误切换到其他板)
    返回:
      obs_by_frame: dict frame -> list[[x,y,z,yaw],...] 候选观测
      tracked:      dict frame -> [x,y,z] 实际关联到的观测 (目标序列)
      filt:         (N,3) 每帧滤波位置
      preds:        (N,4) 每帧 (t, px, py, pz) 对 t+0.25 的预测
    """
    dt = 1.0 / FPS
    kf = CVKalman(dt, q_acc=q_acc, r_xyz=r_xyz) if model == "cv" \
        else CAKalman(dt, q_jerk=q_jerk, r_xyz=r_xyz)
    obs_by_frame = {}
    for r in obs_list:
        obs_by_frame.setdefault(r[0], []).append(r)

    obs_by_frame = {f: [np.array(r[1:5]) for r in v]
                    for f, v in obs_by_frame.items()}

    filt = np.full((total_frames, 3), np.nan)
    tracked = {}
    preds = np.full((total_frames, 4), np.nan)
    preds[:, 0] = np.arange(total_frames) / FPS
    miss = 0
    tracked_yaw = None

    for f in range(total_frames):
        cands = obs_by_frame.get(f)
        if not kf.initialized or miss > max_miss:
            if cands is not None and len(cands):
                j = 0
                kf.init(cands[j][:3])
                tracked_yaw = float(cands[j][3])
                filt[f] = cands[j][:3]
                tracked[f] = cands[j][:3]
                miss = 0
        else:
            kf.predict()
            if cands is not None and len(cands):
                pred_pos = kf.pos()
                best_j, best_d = -1, np.inf
                for j, c in enumerate(cands):
                    d = float(np.linalg.norm(c[:3] - pred_pos))
                    dyaw = abs(angle_diff(c[3], tracked_yaw)) \
                        if tracked_yaw is not None else 0.0
                    if d < best_d and dyaw < yaw_gate:
                        best_d, best_j = d, j
                if best_j >= 0 and best_d < gate:
                    kf.update(cands[best_j][:3])
                    tracked_yaw = float(cands[best_j][3])
                    # 速度幅值限制, 防止错误观测造成发散
                    for i in (1, 3, 5):
                        kf.x[i] = max(-vmax, min(vmax, kf.x[i]))
                    filt[f] = cands[best_j][:3]
                    tracked[f] = cands[best_j][:3]
                    miss = 0
                else:
                    miss += 1
            else:
                miss += 1
            if f not in tracked:
                filt[f] = kf.pos()
        if kf.initialized and miss <= max_miss:
            preds[f, 1:4] = kf.predict_ahead(PRED_HORIZON)
    return obs_by_frame, tracked, filt, preds


def angle_diff(a, b):
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return d


def compare_with_actual(tracked, preds, fps=FPS):
    """预测值 vs 0.25s 后实际观测值 (取目标关联序列)"""
    n_steps = int(round(PRED_HORIZON * fps))
    rows = []      # t, pred_x/y/z, obs_x/y/z
    for f in range(len(preds)):
        if np.isnan(preds[f, 1]):
            continue
        target = f + n_steps
        if target >= len(preds):
            continue
        obs = None
        for d in (0, -1, 1, -2, 2):
            o = tracked.get(target + d)
            if o is not None:
                obs = o
                break
        if obs is None:
            continue
        rows.append((f / fps, preds[f, 1], preds[f, 2], preds[f, 3],
                     obs[0], obs[1], obs[2]))
    arr = np.array(rows)
    if len(arr) == 0:
        return None
    err = arr[:, 4:7] - arr[:, 1:4]
    return arr, err


def plot_results(tracked, filt, preds, arr, err, outdir, model):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    t_all = np.arange(len(filt)) / FPS
    # 观测时间序列 (关联到的目标观测)
    obs_t = np.array(sorted(tracked.keys())) / FPS
    obs_v = np.array([tracked[f] for f in sorted(tracked.keys())])

    # ---------- 图 1: 位置随时间变化 ----------
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, idx, name in zip(axes, range(3), "xyz"):
        ax.plot(obs_t, obs_v[:, idx], ".", ms=3, alpha=0.5, label="观测值")
        ax.plot(t_all, filt[:, idx], lw=1.2, label="卡尔曼滤波值")
        ax.plot(t_all[~np.isnan(preds[:, 1])] + PRED_HORIZON,
                preds[~np.isnan(preds[:, 1]), idx + 1], lw=1.2,
                label="0.25s 预测值(对齐到预测时刻)")
        ax.set_ylabel(f"{name} (m)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title("远距离平移: 装甲板位置随时间变化")
    axes[2].set_xlabel("时间 t (s)")
    save_fig(fig, outdir / "远距离平移_位置时间曲线.png")

    # ---------- 图 2: 俯视轨迹 x-z ----------
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(obs_v[:, 0], obs_v[:, 2], c=obs_t, s=12,
                    cmap="viridis", label="观测轨迹")
    ax.plot(filt[:, 0], filt[:, 2], "r-", lw=1.2, label="滤波轨迹")
    sampled = obs_v[::30]
    ax.quiver(sampled[:-1, 0], sampled[:-1, 2],
              np.diff(sampled[:, 0]), np.diff(sampled[:, 2]),
              angles="xy", scale_units="xy", scale=0.5, color="k", width=0.004)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title("远距离平移: 俯视轨迹 (x-z 平面)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="t (s)")
    save_fig(fig, outdir / "远距离平移_俯视轨迹.png")

    if arr is None:
        return
    # ---------- 图 3: 预测 vs 0.25s 后观测 ----------
    tp, po = arr[:, 0], arr[:, 4:7]
    pp = arr[:, 1:4]
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for ax, idx, name in zip(axes, range(3), "xyz"):
        ax.plot(tp, po[:, idx], ".", ms=3, alpha=0.6, label="0.25s 后实际观测")
        ax.plot(tp, pp[:, idx], "-", lw=1.2, label="0.25s 前预测值")
        ax.set_ylabel(f"{name} (m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_title("远距离平移: 预测值 vs 0.25 s 后实际观测值")
    d = np.linalg.norm(err, axis=1)
    axes[3].plot(tp, d, ".", ms=3)
    axes[3].axhline(np.mean(d), color="r", lw=1, label=f"均值 {np.mean(d):.3f} m")
    axes[3].set_ylabel("3D 误差 (m)"); axes[3].set_xlabel("t (s)")
    axes[3].grid(alpha=0.3); axes[3].legend(fontsize=8)
    save_fig(fig, outdir / "远距离平移_预测对比.png")

    st = error_stats(err)
    print("\n===== 0.25s 预测误差统计 (预测 vs 实际) =====")
    for k, v in st.items():
        print(f"  {k:8s}: {v:.4f}" if isinstance(v, float) else f"  {k:8s}: {v}")
    with open(outdir / "远距离平移_误差统计.txt", "w", encoding="utf-8") as f:
        f.write("0.25 s 预测误差统计 (预测值 vs 0.25 s 后实际观测值)\n")
        for k, v in st.items():
            f.write(f"{k}: {v}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="qA_aim/detections_远距离平移.csv")
    ap.add_argument("--outdir", default="qA_aim/figures")
    ap.add_argument("--pred-csv", default="qA_aim/pred_远距离平移.csv")
    ap.add_argument("--model", default="cv", choices=["cv", "ca"])
    ap.add_argument("--q-acc", type=float, default=4.0)
    ap.add_argument("--r-xyz", type=float, default=0.03)
    ap.add_argument("--q-jerk", type=float, default=50.0)
    ap.add_argument("--gate", type=float, default=0.5)
    args = ap.parse_args()

    rows = sanity_filter(load_detections(args.csv))
    print(f"detections: {len(rows)} valid (sanity filtered)")
    if not rows:
        print("no valid detections!"); return
    total_frames = max(r[0] for r in rows) + 1

    obs_by_frame, tracked, filt, preds = run_tracking(
        rows, total_frames, model=args.model, q_acc=args.q_acc,
        r_xyz=args.r_xyz, q_jerk=args.q_jerk, gate=args.gate)
    arr, err = compare_with_actual(tracked, preds)
    if arr is not None:
        print(f"pred-vs-obs pairs: {len(arr)}")

    # 写出 pred CSV: t 为观测时刻, x,y,z 为 t+0.25 的预测
    write_pred_csv(args.pred_csv,
                   [[t, x, y, z] for t, x, y, z in preds
                    if not np.isnan(x)],
                   ["t", "x", "y", "z"])
    print("pred csv ->", args.pred_csv)
    plot_results(tracked, filt, preds, arr, err, args.outdir, args.model)


if __name__ == "__main__":
    main()
