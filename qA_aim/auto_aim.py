# -*- coding: utf-8 -*-
"""
一体化自瞄软件 auto_aim.py
===========================
将 rm_vision 移植版整合为单一入口的自瞄程序。

流水线 (与真实自瞄系统一致):
  视频帧 -> 装甲板检测 -> PnP 位姿解算 (x,y,z,yaw) -> 目标关联
        -> 卡尔曼滤波 (CV) -> 预测 t+Δt 位置 -> 云台瞄准角 (yaw_g, pitch_g)
        -> 可视化 (装甲板框/预测点/HUD 俯视轨迹/瞄准角) 与 CSV 输出

模式:
  gui     交互式自瞄演示 (默认): 视频回放 + 滑动条实时调参 + HUD + 截图/录像
  demo    无人值守演示: 处理整段视频, 输出带自瞄叠加的演示视频 + 瞄准角 CSV
  detect  批量识别 (等价 run_detect.py)
  predict 平移目标 0.25s 预测 + 曲线 + 误差统计 (等价 run_predict.py)
  gyro    小陀螺车体中心/角速度估计 (等价 run_gyro.py)
  qr      卡尔曼 Q/R 参数分析 (等价 qr_analysis.py)

用法示例:
  python auto_aim.py --mode gui --video <视频> --color red --pose camera_info.json
  python auto_aim.py --mode demo --video <视频> --color red --pose camera_info.json \
      --video-out figures/自瞄演示.mp4 --csv aim_结果.csv
  python auto_aim.py --mode predict --csv qA_aim/detections_远距离平移.csv --q-acc 16 --r-xyz 0.02
  python auto_aim.py --mode gyro --csv qA_aim/detections_中距离陀螺.csv
  python auto_aim.py --mode qr
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.armor_types import RED, BLUE                            # noqa: E402
from src.detector import ArmorParams, Detector, LightParams      # noqa: E402
from src.kalman import CVKalman                                  # noqa: E402
from src.pnp_solver import PnPSolver, load_camera_info           # noqa: E402
from src.video_utils import open_video, VideoWriter              # noqa: E402

FY = 2322.7   # camera_info.json 中的 fy, 用于深度一致性检验


def angle_diff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


# 各视频的推荐检测阈值 (与 run_detect.py 一致)
DETECTOR_PRESETS = {
    ("red", "远距离平移.mp4"): dict(binary_thres=60, min_area=12,
                                    min_brightness=70),
    ("red", "中距离陀螺.mp4"): dict(binary_thres=80, min_area=15,
                                    min_brightness=70),
    ("red", "近距离陀螺.mp4"): dict(binary_thres=80, min_area=15,
                                    min_brightness=70),
    ("blue", "装甲板_蓝.mp4"): dict(binary_thres=90, min_area=15,
                                    min_brightness=0),
}


def preset_for(video_path, color):
    return DETECTOR_PRESETS.get((color, Path(video_path).name), {})


class AimPipeline:
    """自瞄流水线: 检测 -> PnP -> 关联 -> 卡尔曼 -> 预测 -> 瞄准角。

    使用方法 (逐帧):
      st = pipe.process(img, frame_idx)
      st 为状态字典: armors/lights 数量, obs/filt/pred 位置, 瞄准角, 滞后误差等
    """

    def __init__(self, fps, color=RED, camera_info_path=None,
                 det_kwargs=None,
                 q_acc=16.0, r_xyz=0.02, pred_horizon=0.25,
                 gate=0.4, yaw_gate=0.45, max_miss=20, vmax=3.0):
        self.fps = fps
        self.dt = 1.0 / fps
        if not isinstance(color, int):
            color = RED if color == "red" else BLUE
        self.color = color
        self.pred_horizon = pred_horizon

        kw = dict(binary_thres=80, min_brightness=70)
        lp = LightParams()
        lp.min_area = 15
        ap = ArmorParams()
        if color == BLUE:
            kw.update(binary_thres=90, min_brightness=0)
            ap.min_light_ratio = 0.45
            ap.min_small_center_distance = 0.7
            kw["merge_lateral"] = 1.0
        if det_kwargs:
            if "min_area" in det_kwargs:
                lp.min_area = det_kwargs.pop("min_area")
            kw.update(det_kwargs)
        self.det = Detector(detect_color=color, thres_mode="color",
                            light_params=lp, armor_params=ap,
                            merge_segments=True, **kw)

        if camera_info_path is not None:
            self.pnp = PnPSolver(*load_camera_info(camera_info_path))
            self.K = self.pnp.camera_matrix
        else:
            self.pnp = None
            self.K = None

        self.kf = CVKalman(self.dt, q_acc=q_acc, r_xyz=r_xyz)
        self.gate, self.yaw_gate = gate, yaw_gate
        self.max_miss, self.vmax = max_miss, vmax
        self.miss = 0
        self.tracked_yaw = None
        self.history = deque(maxlen=int(2.0 * fps))   # (t, x, z)
        self.preds_at = {}                            # frame -> 预测位置 (t+horizon 时)
        self.lag_errors = []                          # 滞后误差序列 (自检用)
        self.last_state = None

    # ------------------------------------------------------------------
    def set_params(self, q_acc=None, r_xyz=None, pred_horizon=None):
        if q_acc is not None:
            self.kf.q_acc = q_acc
            self.kf.Q = self.kf._build_Q(self.dt, q_acc)
        if r_xyz is not None:
            self.kf.r_xyz = r_xyz
            self.kf.R = np.eye(3) * r_xyz ** 2
        if pred_horizon is not None:
            self.pred_horizon = pred_horizon

    def _candidates(self):
        """检测 + PnP + 深度一致性检验 -> [(pos, yaw)]
        (无相机内参时返回空列表: 只识别不跟踪, 对应蓝色视频场景)"""
        if self.pnp is None:
            return []
        cands = []
        for armor in self.det.armors:
            tvec, yaw = self.pnp.solve(armor)
            if tvec is None:
                continue
            bar_len = (armor.left_light.length + armor.right_light.length) / 2.0
            z_bar = FY * 0.055 / max(bar_len, 1e-6)
            if tvec[2] <= 0.3 or abs(tvec[2] - z_bar) / tvec[2] > 0.35 \
                    or abs(yaw) > 1.5:
                continue
            cands.append((tvec, yaw))
        return cands

    def process(self, img, frame_idx):
        t = frame_idx / self.fps
        self.det.detect(img)
        cands = self._candidates()

        st = {"frame": frame_idx, "t": t,
              "n_lights": len(self.det.lights),
              "n_armors": len(self.det.armors),
              "obs": None, "filt": None, "pred": None,
              "yaw_deg": None, "pitch_deg": None,
              "state": "LOST", "lag_err": None}

        if not self.kf.initialized or self.miss > self.max_miss:
            if cands:
                pos, yaw = cands[0][0], cands[0][1]
                self.kf.init(pos)
                self.tracked_yaw = yaw
                st["obs"] = pos
                st["filt"] = pos
                st["state"] = "TRACKING"
                self.miss = 0
        else:
            self.kf.predict()
            if cands:
                pred_pos = self.kf.pos()
                best_j, best_d = -1, np.inf
                for j, (pos, yaw) in enumerate(cands):
                    d = float(np.linalg.norm(pos - pred_pos))
                    dyaw = abs(angle_diff(yaw, self.tracked_yaw)) \
                        if self.tracked_yaw is not None else 0.0
                    if d < best_d and dyaw < self.yaw_gate:
                        best_d, best_j = d, j
                if best_j >= 0 and best_d < self.gate:
                    pos, yaw = cands[best_j][0], cands[best_j][1]
                    self.kf.update(pos)
                    self.tracked_yaw = yaw
                    for i in (1, 3, 5):
                        self.kf.x[i] = max(-self.vmax,
                                           min(self.vmax, self.kf.x[i]))
                    st["obs"] = pos
                    st["state"] = "TRACKING"
                    self.miss = 0
                else:
                    self.miss += 1
            else:
                self.miss += 1
            st["filt"] = self.kf.pos()
            if self.miss > 0:
                st["state"] = "TEMP_LOST" if self.miss <= self.max_miss \
                    else "LOST"

        if self.kf.initialized and self.miss <= self.max_miss:
            pred = self.kf.predict_ahead(self.pred_horizon)
            st["pred"] = pred
            self.preds_at[frame_idx] = pred
            # 瞄准角: 相机系 x 右/y 下/z 前 -> yaw 绕 y 轴(向左为正),
            # pitch 绕 x 轴(向上为正)
            st["yaw_deg"] = float(np.degrees(np.arctan2(pred[0], pred[2])))
            st["pitch_deg"] = float(np.degrees(
                np.arctan2(-pred[1], np.hypot(pred[0], pred[2]))))
            # 滞后误差自检: 当前观测 vs 0.25s 前的预测
            if st["obs"] is not None:
                past = self.preds_at.get(
                    frame_idx - int(round(self.pred_horizon * self.fps)))
                if past is not None:
                    st["lag_err"] = float(np.linalg.norm(st["obs"] - past))
                    self.lag_errors.append(st["lag_err"])
            self.history.append((t, self.kf.pos()[0], self.kf.pos()[2]))

        self.last_state = st
        return st

    # ------------------------------------------------------------------
    def draw(self, img, st, show_binary=False, show_hud=True):
        """叠加自瞄信息: 装甲板框 + 预测点 + 瞄准角 + HUD 俯视轨迹"""
        out = self.det.draw_results(img, draw_lights=False)
        if st["pred"] is not None and self.K is not None:
            px, py, pz = st["pred"]
            u = int(self.K[0, 0] * px / pz + self.K[0, 2])
            v = int(self.K[1, 1] * py / pz + self.K[1, 2])
            if 0 <= u < out.shape[1] and 0 <= v < out.shape[0]:
                cv2.drawMarker(out, (u, v), (255, 255, 0),
                               cv2.MARKER_CROSS, 40, 2)
                cv2.putText(out, "pred", (u + 12, v - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if st["obs"] is not None:
                    ox, oy, oz = st["obs"]
                    ou = int(self.K[0, 0] * ox / oz + self.K[0, 2])
                    ov = int(self.K[1, 1] * oy / oz + self.K[1, 2])
                    cv2.line(out, (ou, ov), (u, v), (0, 255, 255), 2)
        if show_binary:
            out = np.hstack([out, cv2.cvtColor(self.det.binary_img,
                                               cv2.COLOR_GRAY2BGR)])
        if show_hud:
            out = self._draw_hud(out, st)
        return out

    def _draw_hud(self, img, st):
        """右上角 HUD: 俯视轨迹 (x-z) + 状态文字"""
        H, W = img.shape[:2]
        pw, ph = 230, 200
        x0, y0 = W - pw - 12, 12
        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (30, 30, 30), -1)
        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (80, 80, 80), 1)
        if len(self.history) > 2:
            pts = np.array([(x, z) for _, x, z in self.history])
            xs, zs = pts[:, 0], pts[:, 1]
            cx = (xs.max() + xs.min()) / 2
            span = max(xs.max() - xs.min(), zs.max() - zs.min(), 0.2)
            s = (pw - 30) / (span * 1.2)
            for x, z in pts:
                u = int(x0 + pw / 2 + (x - cx) * s)
                v = int(y0 + ph - 15 - (z - zs.min()) * s)
                cv2.circle(img, (u, v), 1, (0, 200, 0), -1)
            if st["pred"] is not None:
                u = int(x0 + pw / 2 + (st["pred"][0] - cx) * s)
                v = int(y0 + ph - 15 - (st["pred"][2] - zs.min()) * s)
                cv2.drawMarker(img, (u, v), (0, 0, 255),
                               cv2.MARKER_CROSS, 10, 1)
        cv2.putText(img, "top view (x-z)", (x0 + 6, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        lines = [
            f"state: {st['state']}  miss: {self.miss}",
        ]
        if st["obs"] is not None:
            lines.append(f"obs: ({st['obs'][0]:.2f},{st['obs'][2]:.2f})m")
        if st["filt"] is not None:
            lines.append(f"filt: ({st['filt'][0]:.2f},{st['filt'][2]:.2f})m")
        if st["pred"] is not None:
            lines.append(f"pred: ({st['pred'][0]:.2f},{st['pred'][2]:.2f})m")
            lines.append(f"aim: yaw {st['yaw_deg']:+.1f} deg  "
                         f"pitch {st['pitch_deg']:+.1f} deg")
        if st["lag_err"] is not None:
            lines.append(f"lag err: {st['lag_err']:.3f} m")
        y = y0 + ph + 16
        for ln in lines:
            cv2.putText(img, ln, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)
            y += 18
        return img


# ======================================================================
# 模式: demo —— 无人值守演示 (输出演示视频 + 瞄准角 CSV)
# ======================================================================
def mode_demo(args):
    cap, fps, n, size = open_video(args.video)
    fps = args.fps if args.fps is not None else fps
    pipe = AimPipeline(fps, args.color, args.pose,
                       det_kwargs=preset_for(args.video, args.color),
                       q_acc=args.q_acc, r_xyz=args.r_xyz,
                       pred_horizon=args.pred_ms / 1000.0)
    lo, hi = (0, n)
    if args.frames:
        a, _, b = args.frames.partition(":")
        lo, hi = int(a), int(b)
    hi = min(hi, n)
    vw = VideoWriter(args.video_out, fps / args.subsample, size) \
        if args.video_out else None
    rows = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    for f in range(lo, hi):
        ok, img = cap.read()
        if not ok:
            break
        st = pipe.process(img, f)
        if vw is not None and (f - lo) % args.subsample == 0:
            vw.write(pipe.draw(img, st))
        if st["pred"] is not None:
            rows.append([st["t"], st["obs"][0] if st["obs"] is not None else "",
                         st["obs"][1] if st["obs"] is not None else "",
                         st["obs"][2] if st["obs"] is not None else "",
                         st["pred"][0], st["pred"][1], st["pred"][2],
                         st["yaw_deg"], st["pitch_deg"]])
    cap.release()
    if vw:
        vw.close()
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "obs_x", "obs_y", "obs_z",
                        "pred_x", "pred_y", "pred_z", "yaw_deg", "pitch_deg"])
            for r in rows:
                w.writerow([f"{v:.6f}" if isinstance(v, float) else v
                            for v in r])
    lags = [e for e in pipe.lag_errors if e is not None]
    print(f"[demo done] frames {lo}-{hi}, csv -> {args.csv}, "
          f"video -> {args.video_out}")
    if lags:
        print(f"lag error (预测 vs 0.25s 后观测): n={len(lags)}, "
              f"mean={np.mean(lags):.3f} m, rmse="
              f"{np.sqrt(np.mean(np.square(lags))):.3f} m")


# ======================================================================
# 模式: gui —— 交互式自瞄演示
# ======================================================================
def mode_gui(args):
    cap, fps, n, (w, h) = open_video(args.video)
    color = RED if args.color == "red" else BLUE
    pipe = AimPipeline(fps, color, args.pose,
                       det_kwargs=preset_for(args.video, args.color))
    pipe.kf.q_acc, pipe.kf.r_xyz = args.q_acc, args.r_xyz
    pipe.pred_horizon = args.pred_ms / 1000.0

    win = "auto_aim"
    cv2.namedWindow(win)
    tb = [
        ("dom_thres", 60 if color == RED else 90, 255),
        ("min_area", 15, 300),
        ("q_acc x10", int(args.q_acc * 10), 200),
        ("r_xyz x100", int(args.r_xyz * 100), 100),
        ("pred_ms", args.pred_ms, 500),
        ("speed", 3, 20),
        ("view", 0, 1),
        ("frame", 0, n - 1),
    ]
    for name, d, mx in tb:
        cv2.createTrackbar(name, win, d, mx, lambda x: None)

    paused, recording = True, False
    rec_writer = None
    frame_idx = 0
    while True:
        # 读滑动条
        thres = cv2.getTrackbarPos("dom_thres", win)
        area = cv2.getTrackbarPos("min_area", win)
        q = cv2.getTrackbarPos("q_acc x10", win) / 10.0
        r = cv2.getTrackbarPos("r_xyz x100", win) / 100.0
        pred_ms = cv2.getTrackbarPos("pred_ms", win)
        speed = cv2.getTrackbarPos("speed", win)
        view = cv2.getTrackbarPos("view", win)
        slider = cv2.getTrackbarPos("frame", win)

        pipe.det.binary_thres = thres
        pipe.det.l.min_area = area
        pipe.set_params(q_acc=q, r_xyz=r, pred_horizon=pred_ms / 1000.0)
        if slider != frame_idx:
            frame_idx = slider

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        st = pipe.process(img, frame_idx)
        dt_ms = (time.perf_counter() - t0) * 1000
        show = pipe.draw(img, st, show_binary=(view == 1))
        cv2.putText(show, f"f{frame_idx} {dt_ms:.1f}ms "
                          f"{'PAUSED' if paused else 'PLAYING'}"
                          f"{' REC' if recording else ''}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win, show)
        if recording:
            if rec_writer is None:
                rec_writer = VideoWriter("auto_aim_record.mp4", 30, (w, h))
            rec_writer.write(show)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        elif key == 32:
            paused = not paused
        elif key == ord("n"):
            paused = True
            frame_idx = min(frame_idx + 1, n - 1)
            cv2.setTrackbarPos("frame", win, frame_idx)
        elif key == ord("p"):
            paused = True
            frame_idx = max(frame_idx - 1, 0)
            cv2.setTrackbarPos("frame", win, frame_idx)
        elif key == ord("s"):
            cv2.imwrite("auto_aim_screenshot.png", show)
            print("screenshot saved")
        elif key == ord("r"):
            recording = not recording
            if not recording and rec_writer is not None:
                rec_writer.close()
                rec_writer = None
                print("record saved -> auto_aim_record.mp4")
        if not paused:
            frame_idx = (frame_idx + speed) % n
            cv2.setTrackbarPos("frame", win, frame_idx)

    if rec_writer is not None:
        rec_writer.close()
    cap.release()
    cv2.destroyAllWindows()


# ======================================================================
# 模式: menu —— 启动菜单 (bat 双击调用, 中文提示由 Python 输出避免编码问题)
# ======================================================================
def _make_args(video=None, color="red", pose=None):
    return argparse.Namespace(video=video, color=color, pose=pose,
                              q_acc=16.0, r_xyz=0.02, pred_ms=250,
                              video_out=None, csv=None, frames=None,
                              subsample=5, fps=None)


def mode_menu(args):
    data = Path(r"C:\Users\Lenovo\Desktop\给新生 2\data\armor")
    if not data.exists():
        print(f"[警告] 未找到考核数据目录: {data}")
        print("       如数据在别处, 请选 [8] 手动输入视频路径。")
    cam = str(data / "camera_info.json") if data.exists() else None

    while True:
        print()
        print("=" * 46)
        print("  一体化自瞄软件 auto_aim.py - 启动菜单")
        print("=" * 46)
        print("  [1] 交互式自瞄演示 - 远距离平移.mp4 (红)")
        print("  [2] 交互式自瞄演示 - 中距离陀螺.mp4 (红)")
        print("  [3] 交互式自瞄演示 - 近距离陀螺.mp4 (红)")
        print("  [4] 交互式自瞄演示 - 装甲板_蓝.mp4 (蓝, 仅识别)")
        print("  [5] 无人值守演示 - 远距离平移 (演示视频+瞄准角CSV)")
        print("  [6] 批量处理全部视频 (识别/预测/陀螺/Q-R分析)")
        print("  [8] 手动输入视频路径处理")
        print("  [0] 退出")
        print("=" * 46)
        try:
            ch = input("请输入数字后回车: ").strip()
        except EOFError:
            break
        if ch == "0":
            break
        elif ch in ("1", "2", "3", "4") and data.exists():
            video = {"1": "远距离平移.mp4", "2": "中距离陀螺.mp4",
                     "3": "近距离陀螺.mp4", "4": "装甲板_蓝.mp4"}[ch]
            color = "blue" if ch == "4" else "red"
            pose = None if ch == "4" else cam
            a = _make_args(video=str(data / video), color=color, pose=pose)
            mode_gui(a)
        elif ch == "5" and data.exists():
            stem = "远距离平移"
            a = _make_args(video=str(data / "远距离平移.mp4"),
                           color="red", pose=cam)
            a.video_out = f"figures/自瞄演示_{stem}.mp4"
            a.csv = f"aim_{stem}.csv"
            mode_demo(a)
        elif ch == "6" and data.exists():
            for v, c in [("远距离平移.mp4", "red"), ("中距离陀螺.mp4", "red"),
                         ("近距离陀螺.mp4", "red"), ("装甲板_蓝.mp4", "blue")]:
                p = str(data / v)
                pose_args = ["--pose", cam] if c == "red" else []
                _delegate("run_detect",
                          ["--video", p, "--color", c] + pose_args +
                          ["--csv", f"detections_{Path(v).stem}.csv",
                           "--video-out",
                           f"figures/{Path(v).stem}_识别.mp4"])
            _delegate("run_predict",
                      ["--csv", "detections_远距离平移.csv",
                       "--q-acc", "16", "--r-xyz", "0.02"])
            _delegate("run_gyro", ["--csv", "detections_中距离陀螺.csv"])
            _delegate("run_gyro", ["--csv", "detections_近距离陀螺.csv"])
            _delegate("qr_analysis", [])
            print()
            print("批量处理完成! 结果: qA_aim\\*.csv 与 qA_aim\\figures\\")
        elif ch == "8":
            p = input("请输入视频完整路径: ").strip().strip('"')
            if p:
                mode_drop(_make_args(video=p))
        elif ch in ("1", "2", "3", "4", "5", "6"):
            print("[错误] 考核数据目录不存在, 请用 [8] 手动输入视频路径。")
        else:
            print("[无效选项]")


# ======================================================================
# 模式: drop —— 拖拽视频文件处理 (配合 拖拽视频到此处理.bat)
# ======================================================================
def mode_drop(args):
    name = Path(args.video).name
    color = "blue" if ("蓝" in name or "blue" in name.lower()) else "red"
    cam = None
    if color == "red":
        # 红色视频优先使用考核相机内参 (存在才附加)
        cand = Path(r"C:\Users\Lenovo\Desktop\给新生 2\data\armor"
                    r"\camera_info.json")
        if cand.exists():
            cam = str(cand)
    print(f"视频: {args.video}")
    print(f"颜色: {color}  "
          f"{'(使用 camera_info.json 做 PnP)' if cam else '(无内参, 仅识别)'}")
    print("  [1] 交互式自瞄演示 (GUI)")
    print("  [2] 无人值守处理 (演示视频 + 瞄准角 CSV)")
    print("  [3] 仅批量识别 (detections CSV + 识别视频)")
    try:
        ch = input("请选择 (1/2/3, 直接回车默认 2): ").strip() or "2"
    except EOFError:
        ch = "2"
    pose_args = ["--pose", cam] if cam else []
    stem = Path(args.video).stem
    if ch == "1":
        args.color = color
        args.pose = cam
        mode_gui(args)
    elif ch == "3":
        _delegate("run_detect",
                  ["--video", args.video, "--color", color] + pose_args +
                  ["--csv", f"detections_{stem}.csv",
                   "--video-out", f"figures/{stem}_识别.mp4"])
    else:
        args.color = color
        args.pose = cam
        args.video_out = f"figures/自瞄演示_{stem}.mp4"
        args.csv = f"aim_{stem}.csv"
        mode_demo(args)


# ======================================================================
# 模式: diag —— 运行诊断 (配合 运行诊断.bat)
# ======================================================================
def mode_diag(args):
    data = Path(r"C:\Users\Lenovo\Desktop\给新生 2\data\armor")
    print("=" * 46)
    print("  运行诊断")
    print("=" * 46)
    print(f"[1] Python 解释器: {sys.version.split()[0]}")
    print(f"    路径: {sys.executable}")
    here = Path(__file__).resolve().parent
    print(f"[2] 程序目录: {here}")
    print(f"    auto_aim.py: {'OK' if (here / 'auto_aim.py').exists() else '缺失'}")
    print(f"    src/detector.py: "
          f"{'OK' if (here / 'src' / 'detector.py').exists() else '缺失'}")
    print(f"[3] 考核数据目录: {data}")
    print(f"    {'OK' if data.exists() else '缺失 (程序内路径需修改)'}")
    try:
        import cv2
        import matplotlib
        print(f"[4] 依赖库 cv2/numpy/matplotlib: OK (cv2 {cv2.__version__})")
    except ImportError as e:
        print(f"[4] 依赖库缺失: {e}")
        print("    请执行: pip install opencv-python numpy matplotlib")
    print("=" * 46)
    print("若以上全部 OK 仍运行失败, 请把下面命令的完整报错发给我:")
    if data.exists():
        print(f'  python auto_aim.py --mode demo '
              f'--video "{data / "远距离平移.mp4"}" '
              f'--color red --pose "{data / "camera_info.json"}"')
    else:
        print('  python auto_aim.py --mode drop --video "<你的视频路径>"')


# ======================================================================
# 模式: 委托给原脚本 (detect / predict / gyro / qr)
# ======================================================================
def _delegate(module_name, argv):
    import importlib
    mod = importlib.import_module(module_name)
    sys.argv = [module_name] + argv
    mod.main()


def main():
    ap = argparse.ArgumentParser(description="一体化自瞄软件")
    ap.add_argument("--mode", default="gui",
                    choices=["gui", "demo", "drop", "menu", "diag", "detect",
                             "predict", "gyro", "qr"])
    ap.add_argument("--video", default=None)
    ap.add_argument("--color", default="red", choices=["red", "blue"])
    ap.add_argument("--pose", default=None, help="camera_info.json 路径")
    ap.add_argument("--video-out", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--frames", type=str, default=None,
                    help="a:b 只处理帧区间 (demo 模式)")
    ap.add_argument("--subsample", type=int, default=5)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--q-acc", type=float, default=16.0)
    ap.add_argument("--r-xyz", type=float, default=0.02)
    ap.add_argument("--pred-ms", type=int, default=250)
    args = ap.parse_args()

    if args.mode == "gui":
        mode_gui(args)
    elif args.mode == "demo":
        mode_demo(args)
    elif args.mode == "drop":
        mode_drop(args)
    elif args.mode == "menu":
        mode_menu(args)
    elif args.mode == "diag":
        mode_diag(args)
    elif args.mode == "detect":
        argv = ["--video", args.video, "--color", args.color]
        if args.pose:
            argv += ["--pose", args.pose]
        if args.csv:
            argv += ["--csv", args.csv]
        if args.video_out:
            argv += ["--video-out", args.video_out]
        _delegate("run_detect", argv)
    elif args.mode == "predict":
        argv = []
        if args.csv:
            argv += ["--csv", args.csv]
        argv += ["--q-acc", str(args.q_acc), "--r-xyz", str(args.r_xyz)]
        _delegate("run_predict", argv)
    elif args.mode == "gyro":
        _delegate("run_gyro", ["--csv", args.csv] if args.csv else [])
    elif args.mode == "qr":
        _delegate("qr_analysis", [])


if __name__ == "__main__":
    main()
