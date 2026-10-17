# -*- coding: utf-8 -*-
"""
小陀螺目标估计 —— 发挥部分 (2)

对 中距离陀螺.mp4 / 近距离陀螺.mp4:
  读取 detections_*.csv 的 PnP 位姿观测 (x, y, z, yaw)
  -> yaw 扫掠段线性拟合估计旋转角速度 omega
  -> 观测位置中位数估计车体中心
  -> 输出 pred_<视频名>.csv (t, cx, cy, cz, omega) 与对比曲线
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.gyro import estimate_gyro, segment_slopes       # noqa: E402
from src.video_utils import save_fig, setup_chinese_font  # noqa: E402

FPS = 150.0


def load_obs(csv_path, z_tol=0.35):
    """读取 detections CSV -> (frame, x, y, z, yaw) 列表, 含深度检验"""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r["x"] == "" or r["z"] == "":
                continue
            lt = np.array([float(r["lt_x"]), float(r["lt_y"])])
            lb = np.array([float(r["lb_x"]), float(r["lb_y"])])
            bar_len = float(np.linalg.norm(lt - lb))
            z = float(r["z"])
            z_bar = 2322.7 * 0.055 / max(bar_len, 1e-6)
            yaw = float(r["yaw"])
            if z > 0.3 and abs(z - z_bar) / z < z_tol and abs(yaw) < 1.5:
                rows.append((int(float(r["frame"])), float(r["x"]),
                             float(r["y"]), z, yaw))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="qA_aim/figures")
    ap.add_argument("--pred-csv", default=None)
    args = ap.parse_args()

    name = Path(args.csv).name.replace("detections_", "").replace(".csv", "")
    rows = load_obs(args.csv)
    if not rows:
        print("no valid observations"); return
    frames = np.array([r[0] for r in rows], dtype=np.int64)
    t = frames / FPS
    obs = np.array([[r[1], r[2], r[3]] for r in rows])
    yaw = np.array([r[4] for r in rows])
    print(f"{name}: {len(rows)} valid observations, "
          f"t in [{t.min():.2f}, {t.max():.2f}] s")

    res = estimate_gyro(frames, yaw, obs)
    cx, cy, cz = res["cx"], res["cy"], res["cz"]
    r = res["r_est"]
    omega_g = res["omega_global"]
    if omega_g is None:
        print("estimation failed"); return
    print(f"car center: ({cx:.3f}, {cy:.3f}, {cz:.3f}) m, "
          f"radius est: {r:.3f} m")
    print(f"omega_global: {omega_g:.4f} rad/s "
          f"({np.degrees(omega_g):.2f} deg/s, "
          f"{omega_g/(2*np.pi):.3f} rev/s)")
    print(f"segments: {res['n_segments']} "
          f"(strong: {res['n_strong']})")
    segs = res["segs"]
    if len(segs):
        print("  segment slopes (rad/s): " +
              ", ".join(f"{s:+.1f}" for s in sorted(segs[:, 0])))

    # pred CSV: t, cx, cy, cz, omega (每观测帧)
    if args.pred_csv is None:
        args.pred_csv = f"qA_aim/pred_{name}.csv"
    Path(args.pred_csv).parent.mkdir(parents=True, exist_ok=True)
    om_local = res["omega_local"]
    with open(args.pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "cx", "cy", "cz", "omega"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.6f}", f"{cx:.6f}", f"{cy:.6f}",
                        f"{cz:.6f}", f"{om_local[i]:.6f}"])
    print("pred csv ->", args.pred_csv)

    # ---------------- 绘图 ----------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()
    import matplotlib.pyplot as plt

    # 图 1: 观测位置 vs 车体中心估计值
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, idx, nm in zip(axes, range(3), "xyz"):
        ax.plot(t, obs[:, idx], ".", ms=3, alpha=0.6, label="装甲板观测位置")
        center = (cx, cy, cz)[idx]
        ax.axhline(center, color="r", lw=1.5,
                   label=f"车体中心估计 {nm}={center:.3f} m")
        ax.set_ylabel(f"{nm} (m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_title(f"{name}: 装甲板观测位置与车体中心估计值对比")
    axes[2].set_xlabel("时间 t (s)")
    save_fig(fig, outdir / f"{name}_观测vs中心.png")

    # 图 2: 角速度估计曲线
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(t, om_local, lw=1.0, label="逐帧角速度估计(段斜率+平滑)")
    axes[0].axhline(omega_g, color="r", lw=1.5,
                    label=f"全局角速度 {omega_g:.3f} rad/s "
                          f"({omega_g/(2*np.pi):.2f} rev/s)")
    axes[0].set_ylabel("ω (rad/s)"); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{name}: 角速度估计曲线")
    for s, n, tm in segs:
        axes[1].plot([tm - n / FPS / 2, tm + n / FPS / 2], [s, s],
                     color="C0" if abs(s) > 5 else "C7", lw=1.5, alpha=0.7)
    axes[1].axhline(omega_g, color="r", lw=1.2)
    axes[1].set_ylabel("段斜率 (rad/s)"); axes[1].set_xlabel("时间 t (s)")
    axes[1].grid(alpha=0.3)
    axes[1].set_title("各扫掠段斜率 (蓝色: 信息段, 灰色: 噪声段)")
    save_fig(fig, outdir / f"{name}_角速度曲线.png")

    # 图 3: x-z 圆与中心
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(obs[:, 0], obs[:, 2], s=10, c=t, cmap="viridis",
               label="装甲板观测")
    th_fit = np.linspace(0, 2 * np.pi, 200)
    ax.plot(cx + r * np.cos(th_fit), cz + r * np.sin(th_fit), "r--",
            lw=1.5, label=f"旋转圆 r={r:.3f} m")
    ax.plot([cx], [cz], "r*", ms=15, label=f"车体中心 ({cx:.2f}, {cz:.2f})")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title(f"{name}: 装甲板位置与车体中心 (x-z), ω={omega_g:.2f} rad/s")
    ax.legend(); ax.grid(alpha=0.3); ax.set_aspect("equal")
    fig.colorbar(ax.collections[0], ax=ax, label="t (s)")
    save_fig(fig, outdir / f"{name}_圆拟合.png")


if __name__ == "__main__":
    main()
