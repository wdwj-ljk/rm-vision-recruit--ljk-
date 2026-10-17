# -*- coding: utf-8 -*-
"""
装甲板逐帧识别 —— 基本要求 (1)

用法:
  python run_detect.py --video <视频路径> --color red|blue
      [--csv detections_xxx.csv] [--video-out figures/xxx.mp4]
      [--pose camera_info.json] [--subsample 5] [--frames 0:200]
      [--debug] [--thres 80] [--no-merge] [--dump-frame 500]

输出:
  detections_*.csv: frame, lt_x..rt_y (四个关键点), x,y,z,yaw (PnP, 可选)
  识别结果可视化视频 (画装甲板四边形 + 关键点序号, 附位姿文本)
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
from src.pnp_solver import PnPSolver, load_camera_info    # noqa: E402
from src.video_utils import (open_video, VideoWriter,     # noqa: E402
                             write_detections_csv)


def parse_frames(s):
    if not s:
        return None
    a, _, b = s.partition(":")
    return int(a), int(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--color", default="red", choices=["red", "blue"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--video-out", default=None)
    ap.add_argument("--pose", default=None,
                    help="camera_info.json 路径; 提供则做 PnP 解算")
    ap.add_argument("--subsample", type=int, default=5,
                    help="可视化视频每隔几帧写一帧 (150fps->30fps)")
    ap.add_argument("--frames", type=parse_frames, default=None,
                    help="只处理 a:b 帧区间")
    ap.add_argument("--debug", action="store_true",
                    help="打印每帧灯条/装甲板调试信息")
    ap.add_argument("--thres", type=int, default=None)
    ap.add_argument("--min-area", type=float, default=None)
    ap.add_argument("--min-brightness", type=int, default=None)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--dump-frame", type=int, default=None,
                    help="保存指定帧的标注图片")
    args = ap.parse_args()

    color = RED if args.color == "red" else BLUE

    # 各视频的推荐阈值 (在 --thres 未指定时使用)
    presets = {
        ("red", "远距离平移.mp4"): dict(thres=60, min_area=12, min_brightness=70),
        ("red", "中距离陀螺.mp4"): dict(thres=80, min_area=15, min_brightness=70),
        ("red", "近距离陀螺.mp4"): dict(thres=80, min_area=15, min_brightness=70),
        ("blue", "装甲板_蓝.mp4"): dict(thres=90, min_area=15, min_brightness=0),
    }
    vname = Path(args.video).name
    pre = presets.get((args.color, vname),
                      dict(thres=80, min_area=20, min_brightness=80))

    lp = LightParams()
    lp.min_area = args.min_area if args.min_area is not None else pre["min_area"]
    ap_ = ArmorParams()
    if color == BLUE:
        # 蓝色视频: 人员遮挡导致灯条截断, 放宽长度比/中心距约束
        ap_.min_light_ratio = 0.45
        ap_.min_small_center_distance = 0.7
    det = Detector(
        binary_thres=args.thres if args.thres is not None else pre["thres"],
        detect_color=color,
        thres_mode="color",
        min_brightness=(args.min_brightness if args.min_brightness is not None
                        else pre["min_brightness"]),
        light_params=lp, armor_params=ap_,
        merge_segments=not args.no_merge,
        merge_lateral=1.0 if color == BLUE else 0.6)

    pnp = PnPSolver(*load_camera_info(args.pose)) if args.pose else None

    cap, fps, n_frames, (w, h) = open_video(args.video)
    lo, hi = args.frames if args.frames else (0, n_frames)
    hi = min(hi, n_frames)

    vw = None
    if args.video_out:
        Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
        vw = VideoWriter(args.video_out, fps / args.subsample, (w, h))

    rows = []
    n_armor_frames = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    for frame_idx in range(lo, hi):
        ok, img = cap.read()
        if not ok:
            break
        armors = det.detect(img)
        pose_texts = [f"frame {frame_idx}"]
        if armors:
            n_armor_frames += 1
        for armor in armors:
            lt, lb, rb, rt = armor.lt, armor.lb, armor.rb, armor.rt
            x = y = z = yaw = None
            if pnp is not None:
                tvec, yaw_ = pnp.solve(armor)
                if tvec is not None:
                    x, y, z, yaw = tvec[0], tvec[1], tvec[2], yaw_
                    pose_texts.append(
                        f"#{len(pose_texts)} {armor.type} "
                        f"xyz=({x:.2f},{y:.2f},{z:.2f}) yaw={np.degrees(yaw):.1f}")
            rows.append([frame_idx,
                         lt[0], lt[1], lb[0], lb[1], rb[0], rb[1], rt[0], rt[1],
                         x, y, z, yaw])
        if args.debug:
            print(f"--- frame {frame_idx}: {len(det.lights)} lights, "
                  f"{len(armors)} armors")
            for l in det.lights:
                print("   ", l)
            for a in armors:
                print("   ", a)

        if vw is not None and (frame_idx - lo) % args.subsample == 0:
            img_out = det.draw_results(img, pose_texts=pose_texts)
            vw.write(img_out)
        if args.dump_frame is not None and frame_idx == args.dump_frame:
            out_p = Path(args.csv).parent if args.csv else Path(".")
            det.draw_results(img, pose_texts=pose_texts)
            cv2.imwrite(str(out_p / f"annotated_{vname[:-4]}_{frame_idx}.png"),
                        img)

    cap.release()
    if vw:
        vw.close()

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        write_detections_csv(args.csv, rows)

    print(f"[done] {vname}: frames {lo}-{hi}, "
          f"{len(rows)} detections, {n_armor_frames}/{hi - lo} frames with "
          f"armor, csv -> {args.csv}, video -> {args.video_out}")


if __name__ == "__main__":
    main()
