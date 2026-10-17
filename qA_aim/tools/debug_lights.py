# -*- coding: utf-8 -*-
"""调试工具: 转储若干帧的灯条信息 (frame, cx, cy, len, wid, area, ratio) 到 CSV"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.armor_types import RED, BLUE
from src.detector import ArmorParams, Detector, LightParams

VIDEO = Path(r"C:\Users\Lenovo\Desktop\给新生 2\data\armor\远距离平移.mp4")
STEP = 3
MAX_FRAME = 400
THRES = 60

lp = LightParams()
lp.min_area = 12
det = Detector(binary_thres=THRES, detect_color=RED, thres_mode="color",
               min_brightness=70, light_params=lp, armor_params=ArmorParams(),
               merge_segments=True)

cap = cv2.VideoCapture(str(VIDEO))
rows = []
i = 0
while True:
    ok, img = cap.read()
    if not ok or i > MAX_FRAME:
        break
    if i % STEP == 0:
        det.detect(img)
        for l in det.lights:
            rows.append((i, l.center[0], l.center[1], l.length, l.width,
                         l.area, l.ratio))
    i += 1
cap.release()

import csv
out = Path(__file__).parent / "scratch_lights.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "cx", "cy", "len", "wid", "area", "ratio"])
    w.writerows(rows)
print("saved", out, len(rows), "lights")
