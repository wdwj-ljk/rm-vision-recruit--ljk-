# -*- coding: utf-8 -*-
"""调试: 转储候选装甲板 + PnP 位姿, 分析正确配对规律"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.armor_types import RED
from src.detector import ArmorParams, Detector, LightParams
from src.pnp_solver import PnPSolver, load_camera_info

VIDEO = Path(r"C:\Users\Lenovo\Desktop\给新生 2\data\armor\远距离平移.mp4")
CAM = Path(r"C:\Users\Lenovo\Desktop\给新生 2\data\armor\camera_info.json")

lp = LightParams()
lp.min_area = 12
det = Detector(binary_thres=60, detect_color=RED, thres_mode="color",
               min_brightness=70, light_params=lp, armor_params=ArmorParams(),
               merge_segments=True)
pnp = PnPSolver(*load_camera_info(CAM))

cap = cv2.VideoCapture(str(VIDEO))
cap.set(cv2.CAP_PROP_POS_FRAMES, 140)
for i in range(140, 210):
    ok, img = cap.read()
    if not ok:
        break
    det.detect(img)
    if not det.armors:
        continue
    line = f"f{i}: "
    for a in det.armors:
        tvec, yaw = pnp.solve(a)
        if tvec is not None:
            # 由宽度和灯长分别反算 z
            pw = float(np.linalg.norm(a.lb - a.rb))
            z_w = 2320 * (0.23 if a.type == "large" else 0.135) / (pw * 1.0)
            bl = (a.left_light.length + a.right_light.length) / 2
            z_l = 2320 * 0.055 / bl
            line += (f"[{a.type} c=({a.center[0]:.0f},{a.center[1]:.0f}) "
                     f"w={pw:.0f} z_w={z_w:.2f} z_l={z_l:.2f} "
                     f"pose=({tvec[0]:.2f},{tvec[1]:.2f},{tvec[2]:.2f}) "
                     f"yaw={np.degrees(yaw):.1f}] ")
        else:
            line += f"[{a.type} c=({a.center[0]:.0f},{a.center[1]:.0f}) PnPfail] "
    print(line)
cap.release()
