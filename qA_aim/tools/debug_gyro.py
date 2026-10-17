# -*- coding: utf-8 -*-
"""调试: 中距离陀螺候选配对 + PnP 位姿转储"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.armor_types import RED
from src.detector import ArmorParams, Detector, LightParams
from src.pnp_solver import PnPSolver, load_camera_info

VIDEO = Path(r"C:\Users\Lenovo\Desktop\给新生 2\data\armor\中距离陀螺.mp4")
CAM = Path(r"C:\Users\Lenovo\Desktop\给新生 2\data\armor\camera_info.json")

lp = LightParams()
lp.min_area = 15
det = Detector(binary_thres=80, detect_color=RED, thres_mode="color",
               min_brightness=70, light_params=lp, armor_params=ArmorParams(),
               merge_segments=True)
pnp = PnPSolver(*load_camera_info(CAM))

cap = cv2.VideoCapture(str(VIDEO))
cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
for i in range(150, 320, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, img = cap.read()
    if not ok:
        break
    det.detect(img)
    line = f"f{i}: lights={len(det.lights)}"
    for a in det.armors:
        tvec, yaw = pnp.solve(a)
        if tvec is None:
            line += " [PnPfail]"
            continue
        pw = float(np.linalg.norm(a.lb - a.rb))
        z_w_l = 2320 * 0.23 / pw
        z_w_s = 2320 * 0.135 / pw
        bl = (a.left_light.length + a.right_light.length) / 2
        z_l = 2320 * 0.055 / bl
        line += (f" [{a.type} c=({a.center[0]:.0f},{a.center[1]:.0f}) "
                 f"w={pw:.0f} zL={z_w_l:.2f} zS={z_w_s:.2f} zl={z_l:.2f} "
                 f"pose=({tvec[0]:.2f},{tvec[1]:.2f},{tvec[2]:.2f}) "
                 f"yaw={np.degrees(yaw):.0f}]")
    print(line)
cap.release()
