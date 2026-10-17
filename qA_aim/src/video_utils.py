# -*- coding: utf-8 -*-
"""公共工具: 视频读写、CSV 输出、绘图中文支持"""
from __future__ import annotations

import csv
import sys

import cv2
import matplotlib
import numpy as np

# --------------------------------------------------------------------------
# 中文字体 (Windows 自带微软雅黑)
# --------------------------------------------------------------------------
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def setup_chinese_font():
    """配置 matplotlib 中文字体, 成功返回 True"""
    import matplotlib.font_manager as fm
    for path in _FONT_CANDIDATES:
        try:
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            matplotlib.rcParams["font.sans-serif"] = [name,
                                                      "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return False


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------
# 视频
# --------------------------------------------------------------------------


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, n, (w, h)


class VideoWriter:
    def __init__(self, path, fps, size):
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()


def draw_header(img, text):
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

DETECT_HEADER = ["frame", "lt_x", "lt_y", "lb_x", "lb_y",
                 "rb_x", "rb_y", "rt_x", "rt_y",
                 "x", "y", "z", "yaw"]


def write_detections_csv(path, rows):
    """rows: list of dict/list, 与 DETECT_HEADER 对齐; 位姿列可为空"""
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return f"{float(v):.4f}"

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(DETECT_HEADER)
        for r in rows:
            w.writerow([fmt(v) for v in r])


def write_pred_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f"{float(v):.6f}" for v in r])


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------


def error_stats(errors):
    """errors: (N,3) 误差数组; 返回统计 dict"""
    if errors is None or len(errors) == 0:
        return None
    e = np.asarray(errors)
    dist = np.linalg.norm(e, axis=1)
    return {
        "n": len(e),
        "rmse_x": float(np.sqrt(np.mean(e[:, 0] ** 2))),
        "rmse_y": float(np.sqrt(np.mean(e[:, 1] ** 2))),
        "rmse_z": float(np.sqrt(np.mean(e[:, 2] ** 2))),
        "rmse_3d": float(np.sqrt(np.mean(dist ** 2))),
        "mae_x": float(np.mean(np.abs(e[:, 0]))),
        "mae_y": float(np.mean(np.abs(e[:, 1]))),
        "mae_z": float(np.mean(np.abs(e[:, 2]))),
        "mae_3d": float(np.mean(dist)),
        "max_3d": float(np.max(dist)),
    }


def save_fig(fig, path, dpi=130):
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
