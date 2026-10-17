# -*- coding: utf-8 -*-
"""
阈值调节工具 —— 基本要求 (4)

滑动条实时调节检测阈值并显示处理效果:
  dom_thres      颜色差分阈值 (R-max(G,B) / B-max(G,R))
  min_brightness 亮度下限
  min_area       灯条最小面积
  max_angle      灯条最大倾角 (度)
  min_ratio      灯条短边/长边 下界
  max_ratio      灯条短边/长边 上界
  min_light_ratio 配对灯条长度比下界
  frame          帧位置滑动条
  view           0=标注结果 1=二值图 2=左右并排

按键: 空格=播放/暂停, n/p=下一帧/上一帧, s=保存截图, ESC=退出

用法:
  python threshold_tuner.py --video <视频> --color red [--headless --out png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.armor_types import RED, BLUE                    # noqa: E402
from src.detector import (ArmorParams, Detector,          # noqa: E402
                          LightParams)

TRACK = [
    ("dom_thres", 0, 255),
    ("min_brightness", 0, 255),
    ("min_area", 0, 300),
    ("max_angle", 5, 80),
    ("min_ratio", 5, 40),      # x100
    ("max_ratio", 20, 80),     # x100
    ("min_light_ratio", 30, 90),  # x100
    ("frame", 0, 1000),
    ("view", 0, 2),
]


def make_detector(cv2win, color):
    lp = LightParams()
    ap = ArmorParams()
    thres = cv2.getTrackbarPos("dom_thres", cv2win)
    mb = cv2.getTrackbarPos("min_brightness", cv2win)
    lp.min_area = cv2.getTrackbarPos("min_area", cv2win)
    lp.max_angle = cv2.getTrackbarPos("max_angle", cv2win)
    lp.min_ratio = cv2.getTrackbarPos("min_ratio", cv2win) / 100.0
    lp.max_ratio = cv2.getTrackbarPos("max_ratio", cv2win) / 100.0
    ap.min_light_ratio = cv2.getTrackbarPos("min_light_ratio", cv2win) / 100.0
    return Detector(binary_thres=thres, detect_color=color, thres_mode="color",
                    min_brightness=mb, light_params=lp, armor_params=ap,
                    merge_segments=True)


def process_frame(img, color, thres, mb, min_area, max_angle, min_ratio,
                  max_ratio, min_light_ratio):
    lp = LightParams()
    ap = ArmorParams()
    lp.min_area = min_area
    lp.max_angle = max_angle
    lp.min_ratio = min_ratio
    lp.max_ratio = max_ratio
    ap.min_light_ratio = min_light_ratio
    det = Detector(binary_thres=thres, detect_color=color, thres_mode="color",
                   min_brightness=mb, light_params=lp, armor_params=ap,
                   merge_segments=True)
    det.detect(img)
    return det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--color", default="red", choices=["red", "blue"])
    ap.add_argument("--headless", action="store_true",
                    help="无窗口模式: 在若干帧上应用参数并保存截图")
    ap.add_argument("--out", default="qA_aim/figures/threshold_tuner.png")
    ap.add_argument("--frames", default="150,600,1200",
                    help="headless 模式下采样的帧, 逗号分隔")
    args = ap.parse_args()

    color = RED if args.color == "red" else BLUE
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.headless:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        dflt = dict(mb=70, min_area=15, max_angle=40,
                    min_ratio=0.10, max_ratio=0.40, min_light_ratio=0.70) \
            if color == RED else \
            dict(mb=0, min_area=15, max_angle=40,
                 min_ratio=0.10, max_ratio=0.50, min_light_ratio=0.45)
        thres_list = [40, 60, 90] if color == RED else [70, 90, 120]
        panels = []
        for fs in args.frames.split(","):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fs))
            ok, img = cap.read()
            if not ok:
                continue
            row = []
            for thres in thres_list:
                det = process_frame(img, color, thres=thres, **dflt)
                result = det.draw_results(img)
                bin_view = cv2.cvtColor(det.binary_img, cv2.COLOR_GRAY2BGR)
                panel = np.hstack([result, bin_view])
                cv2.putText(panel, f"frame {fs} dom_thres={thres}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2)
                cv2.putText(panel, "left=detection right=binary",
                            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                h, w = panel.shape[:2]
                scale = 640 / w
                if scale < 1:
                    panel = cv2.resize(panel, None, fx=scale, fy=scale)
                row.append(panel)
            panels.append(np.hstack(row))
        if panels:
            out = np.vstack(panels)
            cv2.putText(out, "threshold_tuner: 滑动条调节 dom_thres 的效果对比",
                        (10, out.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(args.out, out)
            print("headless screenshot ->", args.out)
        return

    win = "threshold_tuner"
    cv2.namedWindow(win)
    for name, lo, hi in TRACK:
        default = {"dom_thres": 80 if color == RED else 90,
                   "min_brightness": 70 if color == RED else 0,
                   "min_area": 15, "max_angle": 40,
                   "min_ratio": 10, "max_ratio": 40 if color == RED else 50,
                   "min_light_ratio": 70 if color == RED else 45,
                   "frame": 0, "view": 0}[name]
        cv2.createTrackbar(name, win, default, hi if name != "frame"
                           else n_frames - 1, lambda x: None)

    paused = True
    frame_idx = 0
    while True:
        if not paused:
            frame_idx = (frame_idx + 1) % n_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        if not ok:
            break
        det = make_detector(win, color)
        det.detect(img)
        view = cv2.getTrackbarPos("view", win)
        result = det.draw_results(img)
        if view == 1:
            show = cv2.cvtColor(det.binary_img, cv2.COLOR_GRAY2BGR)
        elif view == 2:
            show = np.hstack([result, cv2.cvtColor(det.binary_img,
                                                   cv2.COLOR_GRAY2BGR)])
        else:
            show = result
        cv2.putText(show, f"frame {frame_idx}  lights={len(det.lights)} "
                          f"armors={len(det.armors)}  "
                          f"{'PAUSED' if paused else 'PLAYING'}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        cv2.imshow(win, show)

        # 帧滑动条 -> 视频帧同步
        slider = cv2.getTrackbarPos("frame", win)
        if slider != frame_idx:
            frame_idx = slider

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        elif key == 32:
            paused = not paused
        elif key == ord("n"):
            paused = True
            frame_idx = (frame_idx + 1) % n_frames
            cv2.setTrackbarPos("frame", win, frame_idx)
        elif key == ord("p"):
            paused = True
            frame_idx = (frame_idx - 1) % n_frames
            cv2.setTrackbarPos("frame", win, frame_idx)
        elif key == ord("s"):
            cv2.imwrite("threshold_tuner_screenshot.png", show)
            print("screenshot saved")
        if not paused:
            cv2.setTrackbarPos("frame", win, frame_idx)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
