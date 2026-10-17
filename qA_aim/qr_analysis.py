# -*- coding: utf-8 -*-
"""
卡尔曼滤波器 Q、R 参数分析 —— 发挥部分 (3)

在 远距离平移.mp4 上, 以 0.25s 预测误差 (预测 vs 实际观测) 为指标:
  1. 固定 R, 扫描 Q (过程噪声: 加速度白噪声方差 q_acc)
  2. 固定 Q, 扫描 R (观测噪声标准差 r_xyz)
  3. 对比 CV / CA 两种模型
输出:
  qr_analysis.csv        各配置误差统计
  Q、R 参数对比图 (误差曲线 + RMSE 柱状图)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_predict import (compare_with_actual, load_detections,   # noqa: E402
                         run_tracking, sanity_filter)
from src.video_utils import error_stats, save_fig, setup_chinese_font  # noqa: E402

CSV = Path("qA_aim/detections_远距离平移.csv")


def main():
    rows = sanity_filter(load_detections(CSV))
    total_frames = max(r[0] for r in rows) + 1

    q_grid = [0.2, 1.0, 4.0, 16.0, 64.0]
    r_grid = [0.01, 0.03, 0.1]
    results = []   # (label, q, r, model, stats, err3d, t)
    for q in q_grid:
        for r in r_grid:
            obs, tracked, filt, preds = run_tracking(rows, total_frames,
                                                     model="cv", q_acc=q,
                                                     r_xyz=r)
            arr, err = compare_with_actual(tracked, preds)
            st = error_stats(err) if err is not None else {"rmse_3d": np.nan}
            results.append((f"CV q={q} r={r}", q, r, "cv", st,
                            np.linalg.norm(err, axis=1) if err is not None
                            else None, arr[:, 0] if arr is not None else None))
            print(f"CV  q_acc={q:5.1f} r_xyz={r:.2f}: "
                  f"RMSE3d={st['rmse_3d']:.4f} m, n={st['n']}")

    # CA 模型对照
    for r in [0.03]:
        obs, tracked, filt, preds = run_tracking(rows, total_frames,
                                                 model="ca", r_xyz=r,
                                                 q_jerk=50.0)
        arr, err = compare_with_actual(tracked, preds)
        st = error_stats(err) if err is not None else {"rmse_3d": np.nan}
        results.append((f"CA q_jerk=50 r={r}", np.nan, r, "ca", st,
                        np.linalg.norm(err, axis=1) if err is not None
                        else None, arr[:, 0] if arr is not None else None))
        print(f"CA  q_jerk=50 r_xyz={r:.2f}: RMSE3d={st['rmse_3d']:.4f} m")

    # 保存 CSV
    import csv
    with open("qA_aim/qr_analysis.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "model", "q_acc", "r_xyz", "rmse_x", "rmse_y",
                    "rmse_z", "rmse_3d", "mae_3d", "max_3d", "n"])
        for label, q, r, model, st, _, _ in results:
            w.writerow([label, model, q, r,
                        st.get("rmse_x"), st.get("rmse_y"), st.get("rmse_z"),
                        st.get("rmse_3d"), st.get("mae_3d"), st.get("max_3d"),
                        st.get("n")])

    setup_chinese_font()
    import matplotlib.pyplot as plt

    # ---- 图 1: 固定 R=0.03 扫 Q, 误差随时间曲线 ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for label, q, r, model, st, err3d, t in results:
        if model == "cv" and r == 0.03:
            axes[0].plot(t, err3d, ".", ms=2.5, alpha=0.6,
                         label=f"q_acc={q}  (RMSE3d={st['rmse_3d']:.3f} m)")
    axes[0].set_ylabel("预测-实际 3D 误差 (m)")
    axes[0].set_title("固定 R=0.03: 过程噪声 Q (q_acc) 对 0.25s 预测误差的影响")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    for label, q, r, model, st, err3d, t in results:
        if model == "cv" and q == 4.0:
            axes[1].plot(t, err3d, ".", ms=2.5, alpha=0.6,
                         label=f"r_xyz={r}  (RMSE3d={st['rmse_3d']:.3f} m)")
    axes[1].set_ylabel("预测-实际 3D 误差 (m)")
    axes[1].set_xlabel("t (s)")
    axes[1].set_title("固定 Q (q_acc=4): 观测噪声 R (r_xyz) 对 0.25s 预测误差的影响")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    save_fig(fig, "qA_aim/figures/QR分析_误差曲线.png")

    # ---- 图 2: RMSE 柱状图 ----
    fig, ax = plt.subplots(figsize=(12, 5.5))
    labels = [r[0] for r in results]
    vals = [r[4]["rmse_3d"] for r in results]
    colors = ["#4477aa"] * len(vals)
    for i, r in enumerate(results):
        if r[3] == "ca":
            colors[i] = "#dd7733"
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("0.25s 预测 RMSE (3D, m)")
    ax.set_title("不同 Q、R 配置与模型 (CV/CA) 的 0.25s 预测误差对比")
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                ha="center", fontsize=8)
    save_fig(fig, "qA_aim/figures/QR分析_柱状图.png")
    print("figures saved -> qA_aim/figures/QR分析_*.png")
    print("csv -> qA_aim/qr_analysis.csv")


if __name__ == "__main__":
    main()
