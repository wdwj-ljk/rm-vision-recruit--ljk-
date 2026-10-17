# -*- coding: utf-8 -*-
"""
针孔相机模型内参标定 + 基于 PnP 的平面目标测距
=================================================

功能
----
1. 标定（张正友棋盘格法）
   - 采集 15~25 张不同位置、不同姿态的棋盘格图像（或读取已有照片）
   - 输出内参矩阵 K、畸变系数 dist、平均/逐张重投影误差
   - 自动选取一张代表性图像，输出去畸变前后对比图 undistort_comparison.jpg
2. 测距（cv2.solvePnP）
   - 以 A4 纸（210 mm x 297 mm）等已知尺寸平面目标为参照
   - 自动检测 4 个角点（Otsu/Canny 提取最大四边形），失败时可鼠标手动点选
   - 求解位姿，输出相机到目标中心的距离、深度及角点重投影误差

针孔模型
--------
    s * [u, v, 1]^T = K * [R | t] * [X, Y, Z, 1]^T
    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    dist = [k1, k2, p1, p2, k3]

用法
----
1) 摄像头采集 20 张并标定（按 SPACE 保存，Q 提前结束；不加数字默认 20 张，建议 15~25）:
    python camera_calibration.py --capture --grid 9 6 --square 25
2) 用已有照片标定:
    python camera_calibration.py --folder ./imgs --pattern "*.jpg" --grid 9 6 --square 25
3) 标定完成后，测量照片中 A4 纸到相机的距离:
    python camera_calibration.py --measure photo.jpg
    # 自动检测失败时手动点选 4 个角点（顺序: 左上 -> 右上 -> 右下 -> 左下）:
    python camera_calibration.py --measure photo.jpg --manual
    # 直接抓取一帧实时测量:
    python camera_calibration.py --live
4) 用标定参数对任意照片去畸变 / 生成对比图:
    python camera_calibration.py --undistort photo.jpg
    python camera_calibration.py --compare photo.jpg
5) 合成数据自检（无需相机/照片，验证标定与测距全流程）:
    python camera_calibration.py --demo

拍摄建议
--------
- 棋盘格打印在平整硬板上，单格边长已知（单位与 --square 一致，建议毫米）
- 拍摄 15~25 张：覆盖画面不同位置、不同角度（绕各轴倾斜 20~40 度）、不同距离
- 避免运动模糊与反光；角点必须完整出现在画面内
- 标定质量看"平均重投影误差"，一般应 < 0.5 像素
- 测距时尽量让 A4 纸占画面足够面积、边缘清晰、与背景对比明显

依赖: numpy, opencv-python
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

A4_SIZE_MM = (210.0, 297.0)


# ===========================================================================
# 一、棋盘格标定
# ===========================================================================

def build_object_points(grid, square_size_mm):
    """生成棋盘平面上 (Z=0) 的内角点三维坐标（行优先，与检测顺序一致）。"""
    objp = np.zeros((grid[0] * grid[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid[0], 0:grid[1]].T.reshape(-1, 2) * square_size_mm
    return objp


def find_corners(img, grid, subpix=True):
    """检测棋盘格内角点。返回 (是否成功, 角点 Nx1x2 数组)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, grid, flags)
    if ret and subpix:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners


def run_calibration(images, labels, grid, square_size_mm, out_dir=None):
    """对一组图像执行张正友标定，返回 (K, dist, 平均重投影误差)。
    同时自动选取一张代表性图像生成去畸变前后对比图。"""
    objp = build_object_points(grid, square_size_mm)
    objpoints, imgpoints, good_imgs, good_labels = [], [], [], []

    for img, lab in zip(images, labels):
        ok, corners = find_corners(img, grid)
        if ok:
            objpoints.append(objp)
            imgpoints.append(corners)
            good_imgs.append(img)
            good_labels.append(lab)
        else:
            print(f"  [跳过] 未检测到全部角点: {lab}")

    if len(good_imgs) < 3:
        raise RuntimeError("有效图片不足 3 张：请从不同角度、不同距离多拍几张棋盘格")

    h, w = good_imgs[0].shape[:2]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None)

    # 注意: cv2 5.x 中角点可能是 (N,2)、投影结果是 (N,1,2)，
    # 直接相减会被广播成 (N,N,2)（变成全组合距离），必须先统一 reshape
    per_img = []
    for op, ip, rv, tv in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        e = np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - ip.reshape(-1, 2)) ** 2, axis=1)))
        per_img.append(float(e))

    mean_err = float(np.mean(per_img))
    print("\n===== 标定结果 =====")
    print(f"有效图片: {len(good_imgs)} 张, 图像尺寸: {w} x {h}")
    print("内参矩阵 K:\n", K)
    print(f"焦距: fx = {K[0, 0]:.2f} px, fy = {K[1, 1]:.2f} px")
    print(f"主点: cx = {K[0, 2]:.2f} px, cy = {K[1, 2]:.2f} px")
    print(f"畸变系数 [k1, k2, p1, p2, k3]: {dist.ravel()}")
    print(f"平均重投影误差: {mean_err:.4f} px  (rms={rms:.4f})")
    for lab, e in zip(good_labels, per_img):
        print(f"  {lab}: {e:.4f} px")

    if out_dir:
        save_results(out_dir, K, dist, mean_err, per_img, good_labels, (w, h))
        # 选取重投影误差最接近中位数的一张做去畸变前后对比
        rep_idx = int(np.argmin(np.abs(np.array(per_img) - np.median(per_img))))
        cmp_path = os.path.join(out_dir, "undistort_comparison.jpg")
        make_undistort_comparison(good_imgs[rep_idx], K, dist, cmp_path)
        print(f"已选取代表性图像 {good_labels[rep_idx]} 生成去畸变前后对比图: {cmp_path}")
    return K, dist, mean_err


def save_results(out_dir, K, dist, mean_err, per_img, labels, image_size):
    """把内参保存为 intrinsics.json（人可读）与 intrinsics.npz（程序加载）。"""
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "model": "pinhole",
        "image_size": list(image_size),
        "camera_matrix": K.tolist(),
        "distortion": dist.ravel().tolist(),
        "mean_reprojection_error_px": mean_err,
        "per_image_reprojection_error_px": per_img,
        "images": labels,
    }
    with open(os.path.join(out_dir, "intrinsics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    np.savez(os.path.join(out_dir, "intrinsics.npz"),
             K=K, dist=dist, image_size=np.array(image_size))
    print(f"\n参数已保存到 {out_dir} 目录 (intrinsics.json / intrinsics.npz)")


def load_intrinsics(out_dir):
    """读取已保存的标定结果。"""
    p = os.path.join(out_dir, "intrinsics.npz")
    if not os.path.isfile(p):
        raise RuntimeError(f"找不到标定结果 {p}，请先完成标定")
    d = np.load(p)
    return d["K"], d["dist"]


# ===========================================================================
# 二、去畸变与对比图
# ===========================================================================

def make_undistort_comparison(img, K, dist, out_path):
    """生成去畸变前后并排对比图（各叠画两条水平参考线）。"""
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1)
    undist = cv2.undistort(img, K, dist, None, newK)
    y0, y1 = int(h * 0.25), int(h * 0.75)
    for im in (img, undist):
        cv2.line(im, (0, y0), (w - 1, y0), (0, 255, 0), 2)
        cv2.line(im, (0, y1), (w - 1, y1), (0, 255, 0), 2)
    side = np.hstack([img, undist])
    cv2.putText(side, "Original", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(side, "Undistorted", (w + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imwrite(out_path, side)
    return side


def undistort_image(img_path, out_path, K, dist):
    """使用标定好的参数对一张照片去畸变。"""
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"无法读取图片: {img_path}")
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, K, dist, None, newK)
    cv2.imwrite(out_path, dst)
    print(f"已保存去畸变图像: {out_path}")


# ===========================================================================
# 三、PnP 测距（A4 纸等已知尺寸平面目标）
# ===========================================================================

def order_points(pts):
    """把无序 4 点排序为 [左上, 右上, 右下, 左下]。"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]      # 左上: x + y 最小
    rect[2] = pts[np.argmax(s)]      # 右下: x + y 最大
    rect[1] = pts[np.argmin(diff)]   # 右上: x - y 最小
    rect[3] = pts[np.argmax(diff)]   # 左下: x - y 最大
    return rect


def detect_paper_quad(img, min_area_ratio=0.05):
    """自动检测图像中最大的四边形（纸张）。返回排序后的 4x2 角点，失败返回 None。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    img_area = h * w
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 方案 A：Otsu 二值化（纸张比背景亮或暗两种极性都试）
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 方案 B：Canny 边缘
    med = float(np.median(blur))
    edges = cv2.Canny(blur, max(10.0, 0.66 * med), max(30.0, 1.33 * med))
    masks = [bw, 255 - bw, edges]

    candidates = []
    for mask in masks:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area_ratio * img_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            pts = approx.reshape(4, 2)
            if np.any(pts < 8) or np.any(pts > np.array([w - 8, h - 8])):
                continue  # 贴边，角点不可靠
            candidates.append((area, pts))
    if not candidates:
        return None

    candidates.sort(key=lambda t: -t[0])
    quad = order_points(candidates[0][1])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    quad = cv2.cornerSubPix(gray, quad.reshape(-1, 1, 2), (11, 11), (-1, -1),
                            criteria).reshape(4, 2)
    return quad


def click_paper_corners(img):
    """鼠标手动点选 4 个角点，顺序：左上 -> 右上 -> 右下 -> 左下。"""
    pts = []
    h, w = img.shape[:2]
    max_disp = 1280
    s = min(1.0, max_disp / max(h, w))
    disp = cv2.resize(img, (int(w * s), int(h * s)))
    win = "A4 corners: click 1=TL 2=TR 3=BR 4=BL (ESC=cancel)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / s, y / s))
            cv2.circle(disp, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(disp, str(len(pts)), (x + 12, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow(win, disp)

    print("请按顺序点击 A4 纸的 4 个角点: 1 左上, 2 右上, 3 右下, 4 左下 (ESC 取消)")
    cv2.imshow(win, disp)
    cv2.setMouseCallback(win, on_mouse)
    while len(pts) < 4:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q')):
            break
    cv2.destroyWindow(win)
    cv2.waitKey(1)
    if len(pts) < 4:
        raise RuntimeError("未完成 4 个角点的选取")
    return np.array(pts, dtype=np.float32)


def solve_pnp_distance(corners, K, paper_size_mm):
    """4 个平面角点 -> cv2.solvePnP -> 相机到目标中心距离 (mm) 等。"""
    W, H = paper_size_mm
    objp = np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], np.float32)
    imgpts = corners.reshape(-1, 1, 2).astype(np.float32)

    # 平面目标用 IPPE（对共面点最稳），会返回两个解，取目标在相机前方 (tz > 0) 的
    res = cv2.solvePnPGeneric(objp, imgpts, K, None, flags=cv2.SOLVEPNP_IPPE)
    rvecs, tvecs = res[1], res[2]
    best = None
    for rv, tv in zip(rvecs, tvecs):
        if tv[2, 0] > 0:
            best = (rv, tv)
            break
    if best is None:
        best = (rvecs[0], tvecs[0])
    rvec, tvec = best

    # 以 IPPE 结果为初值，迭代精化
    _, rvec, tvec = cv2.solvePnP(
        objp, imgpts, K, None, rvec, tvec,
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

    R, _ = cv2.Rodrigues(rvec)
    center_obj = np.array([[W / 2], [H / 2], [0.0]])
    center_cam = R @ center_obj + tvec          # 目标中心在相机坐标系下的位置
    distance_mm = float(np.linalg.norm(center_cam))
    depth_mm = float(tvec[2, 0])

    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, None)
    err_px = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))
    return {
        "rvec": rvec, "tvec": tvec, "R": R,
        "distance_mm": distance_mm, "depth_mm": depth_mm,
        "reproj_err_px": err_px,
    }


def annotate_pnp(img, corners, K, res):
    """在图上画出角点、物体坐标系与测距结果。"""
    disp = img.copy()
    cv2.polylines(disp, [corners.astype(np.int32)], True, (0, 255, 0), 3)
    for i, (x, y) in enumerate(corners):
        cv2.circle(disp, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(disp, str(i + 1), (int(x) + 12, int(y) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # 以角点 1 为原点画 100 mm 长的物体坐标系: X 红, Y 绿, Z 蓝
    axis_obj = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], np.float32)
    axis_img, _ = cv2.projectPoints(axis_obj, res["rvec"], res["tvec"], K, None)
    o = tuple(axis_img[0].ravel().astype(int))
    cv2.line(disp, o, tuple(axis_img[1].ravel().astype(int)), (0, 0, 255), 3)
    cv2.line(disp, o, tuple(axis_img[2].ravel().astype(int)), (0, 255, 0), 3)
    cv2.line(disp, o, tuple(axis_img[3].ravel().astype(int)), (255, 0, 0), 3)
    cv2.putText(disp, f"D = {res['distance_mm'] / 1000.0:.3f} m", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
    return disp


def measure(img, K, dist, paper_size_mm=A4_SIZE_MM, manual=False,
            allow_manual=True, min_area_ratio=0.05, quiet=False):
    """对单张图像完成平面目标测距，返回结果 dict（含 annotated 标注图）。"""
    h, w = img.shape[:2]
    if dist is not None and np.any(dist):
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        img_u = cv2.undistort(img, K, dist, None, newK)
        K_use = newK
    else:
        img_u, K_use = img, K

    if manual:
        corners = click_paper_corners(img_u)
    else:
        corners = detect_paper_quad(img_u, min_area_ratio)
        if corners is None:
            if allow_manual:
                print("自动检测失败，改为手动点选 4 个角点")
                corners = click_paper_corners(img_u)
            else:
                raise RuntimeError("自动检测 A4 纸张失败")

    res = solve_pnp_distance(corners, K_use, paper_size_mm)
    res["annotated"] = annotate_pnp(img_u, corners, K_use, res)

    if not quiet:
        print("\n===== A4 纸 PnP 测距 =====")
        print(f"目标尺寸: {paper_size_mm[0]:.0f} x {paper_size_mm[1]:.0f} mm")
        print("角点像素坐标 (左上/右上/右下/左下):\n", corners)
        print(f"相机到纸张中心距离: {res['distance_mm']:.1f} mm = {res['distance_mm'] / 1000:.3f} m")
        print(f"深度 tz: {res['depth_mm']:.1f} mm")
        print(f"平移向量 tvec (mm): {res['tvec'].ravel()}")
        print(f"旋转向量 rvec: {res['rvec'].ravel()}")
        print(f"角点重投影误差: {res['reproj_err_px']:.3f} px")
    return res


def measure_file(img_path, K, dist, out_dir, paper_size_mm, manual=False):
    """读取图片 -> 测距 -> 保存标注结果图。"""
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"无法读取图片: {img_path}")
    res = measure(img, K, dist, paper_size_mm, manual=manual)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "pnp_measure.jpg")
    cv2.imwrite(out, res["annotated"])
    print(f"标注结果已保存: {out}")
    return res


# ===========================================================================
# 四、摄像头采集
# ===========================================================================

def capture_images(n, cam_id, folder):
    """打开摄像头，按 SPACE 保存当前帧，按 Q/ESC 提前结束。"""
    os.makedirs(folder, exist_ok=True)
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 (编号 {cam_id})")
    saved, idx = 0, 0
    print(f"按 SPACE 保存当前帧，按 Q 提前结束，共需 {n} 张")
    print("提示：边移动/旋转棋盘格边拍，覆盖画面各区域、不同倾角与距离")
    while saved < n:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("capture - SPACE: save, Q: quit", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            p = os.path.join(folder, f"frame_{idx:03d}.png")
            cv2.imwrite(p, frame)
            print(f"[{saved + 1}/{n}] 已保存 {p}")
            saved += 1
            idx += 1
        elif key in (ord('q'), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    return folder


def grab_frame(cam_id, out_dir):
    """抓取一帧（用于 --live 实时测距），按 SPACE 抓取。"""
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 (编号 {cam_id})")
    print("将 A4 纸正对相机放好，按 SPACE 抓取当前帧，按 Q 取消")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("press SPACE to grab, Q to quit", frame)
        k = cv2.waitKey(30) & 0xFF
        if k == ord(' '):
            os.makedirs(out_dir, exist_ok=True)
            p = os.path.join(out_dir, "live_frame.jpg")
            cv2.imwrite(p, frame)
            cap.release()
            cv2.destroyAllWindows()
            return p
        if k in (ord('q'), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    raise RuntimeError("已取消抓取")


# ===========================================================================
# 五、合成数据自检（无需相机/照片）
# ===========================================================================

def render_board(grid, square_mm, scale=40, margin=80):
    """按物理尺寸画一张棋盘图（1 mm = scale 像素），四周留白边。"""
    rows, cols = grid[1] + 1, grid[0] + 1       # 方块数 = 内角点数 + 1
    side = int(round(square_mm * scale))
    w, h = cols * side + 2 * margin, rows * side + 2 * margin
    board = np.full((h, w), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:                 # 左上角为黑色（检测器偏好朝向）
                board[margin + r * side: margin + (r + 1) * side,
                      margin + c * side: margin + (c + 1) * side] = 0
    return board


def render_paper(paper_mm, scale=4, margin=80):
    """按物理尺寸画一张"白纸"图（深色背景，1 mm = scale 像素）。"""
    W, H = paper_mm
    w, h = int(W * scale) + 2 * margin, int(H * scale) + 2 * margin
    src = np.full((h, w), 45, np.uint8)
    src[margin:margin + int(H * scale), margin:margin + int(W * scale)] = 235
    return src


def demo(grid=(9, 6), square_mm=25.0, n_images=15, out_dir=None, seed=0):
    """合成数据自检：1) 棋盘标定对比真值; 2) A4 PnP 测距对比真值。"""
    print("=== 合成数据自检 ===")
    if out_dir:
        print(f"注意：自检结果会写入 {out_dir} 目录，覆盖其中的真实标定参数")
    rng = np.random.default_rng(seed)
    W_img, H_img = 1280, 960
    K_true = np.array([[950.0, 0.0, W_img / 2],
                       [0.0, 950.0, H_img / 2],
                       [0.0, 0.0, 1.0]])

    # ---- Part 1: 棋盘格标定自检 ----
    print("\n[1/2] 棋盘格标定自检")
    scale, margin = 40, 80
    board = render_board(grid, square_mm, scale, margin)
    variants = [board, 255 - board]
    Ts = np.array([[1, 0, -margin], [0, 1, -margin], [0, 0, scale]], float) / scale

    images, labels = [], []
    tries = 0
    while len(images) < n_images and tries < 500:
        tries += 1
        rvec = rng.uniform(-0.45, 0.45, (3, 1))
        tvec = np.array([[rng.uniform(-200, 200)],
                         [rng.uniform(-150, 150)],
                         [rng.uniform(500, 1100)]], float)
        R, _ = cv2.Rodrigues(rvec)
        H_pix = K_true @ np.hstack([R[:, :2], tvec]) @ Ts
        for src in variants:
            img = cv2.warpPerspective(src, H_pix, (W_img, H_img),
                                      flags=cv2.INTER_LINEAR, borderValue=255)
            ok, _ = find_corners(img, grid)
            if ok:
                images.append(img)
                labels.append(f"synthetic_{len(images):02d}")
                break
    if len(images) < 3:
        raise RuntimeError("合成棋盘图角点检测失败，请检查 OpenCV 环境")

    K_est, dist_est, mean_err = run_calibration(images, labels, grid, square_mm, out_dir)
    print("\n---- 与真值对比 ----")
    print("真值 K:\n", K_true)
    print("估计 K:\n", K_est)
    print(f"K 的最大绝对误差: {np.abs(K_est - K_true).max():.3f} px")
    print(f"平均重投影误差: {mean_err:.4f} px")
    ok1 = np.abs(K_est - K_true).max() < 10
    print("标定自检通过" if ok1 else "标定自检未通过：误差偏大")

    # ---- Part 2: A4 PnP 测距自检 ----
    print("\n[2/2] A4 纸 PnP 测距自检")
    paper = A4_SIZE_MM
    scale2, margin2 = 4, 80
    paper_img = render_paper(paper, scale2, margin2)
    Ts2 = np.array([[1, 0, -margin2], [0, 1, -margin2], [0, 0, scale2]], float) / scale2
    targets = [650.0, 900.0, 1100.0]     # 三个不同的目标距离 (mm)
    max_err = 0.0
    last_annotated = None
    n_ok = 0
    for i, tz in enumerate(targets):
        res = None
        for attempt in range(30):        # 位姿可能把纸渲染出画面，重采样直到可检测
            rvec = np.array([[rng.uniform(-0.25, 0.25)],
                             [rng.uniform(-0.25, 0.25)],
                             [rng.uniform(-0.30, 0.30)]], float)
            tvec = np.array([[rng.uniform(-100, 100)],
                             [rng.uniform(-130, 130)],
                             [tz]], float)
            R, _ = cv2.Rodrigues(rvec)
            Hp = K_true @ np.hstack([R[:, :2], tvec]) @ Ts2
            img = cv2.warpPerspective(paper_img, Hp, (W_img, H_img), borderValue=45)
            img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape),
                          0, 255).astype(np.uint8)
            try:
                res = measure(img, K_est, dist_est, paper, allow_manual=False,
                              min_area_ratio=0.02, quiet=True)
                break
            except RuntimeError:
                res = None
        if res is None:
            print(f"  目标距离 {tz:.0f} mm: 未能生成可检测位姿（跳过）")
            continue
        c_obj = np.array([[paper[0] / 2], [paper[1] / 2], [0.0]])
        gt = float(np.linalg.norm(R @ c_obj + tvec))
        err = abs(res["distance_mm"] - gt)
        max_err = max(max_err, err)
        last_annotated = res["annotated"]
        n_ok += 1
        print(f"  位姿 {i + 1}: 真值 {gt:8.1f} mm | 测量 {res['distance_mm']:8.1f} mm | 误差 {err:6.2f} mm")
    if last_annotated is not None and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "pnp_measure_demo.jpg"), last_annotated)
    if n_ok == 0:
        raise RuntimeError("A4 合成图全部检测失败，请检查环境")
    rel = max_err / np.mean(targets) * 100
    print(f"最大测距误差: {max_err:.2f} mm (相对 {rel:.2f}%)")
    print("PnP 测距自检通过" if max_err < 15 else "PnP 测距自检未通过")
    return K_est


# ===========================================================================
# 命令行入口
# ===========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="针孔相机模型内参标定 + PnP 平面目标测距",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--folder", help="棋盘格照片所在目录")
    ap.add_argument("--pattern", default="*.jpg", help="照片匹配模式，如 *.png")
    ap.add_argument("--grid", nargs=2, type=int, default=(9, 6),
                    metavar=("COLS", "ROWS"), help="棋盘格内角点数量（列 行）")
    ap.add_argument("--square", type=float, default=25.0,
                    help="棋盘格单格边长（毫米）。只影响外参尺度，不影响内参")
    ap.add_argument("--out", default="calib_out", help="标定结果输出目录")
    ap.add_argument("--capture", nargs="?", type=int, const=20, default=None,
                    metavar="N", help="用摄像头现拍 N 张后标定（默认 20 张，建议 15~25）")
    ap.add_argument("--camera", type=int, default=0, help="摄像头编号")
    ap.add_argument("--demo", action="store_true", help="合成数据自检（无需相机/照片）")
    ap.add_argument("--undistort", metavar="IMG", help="用已标定参数校正一张图片")
    ap.add_argument("--compare", metavar="IMG", help="生成一张图片的去畸变前后对比图")
    ap.add_argument("--measure", metavar="IMG", help="测量图片中 A4 纸到相机的距离（PnP）")
    ap.add_argument("--manual", action="store_true",
                    help="与 --measure/--live 连用：手动点选 4 个角点")
    ap.add_argument("--paper", nargs=2, type=float, default=list(A4_SIZE_MM),
                    metavar=("W_MM", "H_MM"), help="目标平面尺寸（毫米），默认 A4 = 210 297")
    ap.add_argument("--live", action="store_true", help="抓取一帧实时测量 A4 纸距离")
    args = ap.parse_args(argv)

    if args.demo:
        demo(grid=tuple(args.grid), square_mm=args.square, out_dir=args.out)
        return

    if args.undistort:
        K, dist = load_intrinsics(args.out)
        out_img = os.path.splitext(args.undistort)[0] + "_undistorted.jpg"
        undistort_image(args.undistort, out_img, K, dist)
        return

    if args.compare:
        K, dist = load_intrinsics(args.out)
        img = cv2.imread(args.compare)
        if img is None:
            raise RuntimeError(f"无法读取图片: {args.compare}")
        os.makedirs(args.out, exist_ok=True)
        out = os.path.join(args.out, "undistort_comparison.jpg")
        make_undistort_comparison(img, K, dist, out)
        print(f"对比图已保存: {out}")
        return

    if args.measure or args.live:
        K, dist = load_intrinsics(args.out)
        if args.live:
            img_path = grab_frame(args.camera, args.out)
        else:
            img_path = args.measure
        measure_file(img_path, K, dist, args.out, tuple(args.paper), manual=args.manual)
        return

    # 标定流程
    if args.capture:
        folder = capture_images(args.capture, args.camera, args.folder or "captured")
    else:
        folder = args.folder
        if not folder:
            ap.error("请提供 --folder（已有照片目录）、--capture [N]（摄像头现拍）或 --demo（自检）")
    paths = sorted(glob.glob(os.path.join(folder, args.pattern)))
    if not paths:
        raise RuntimeError(f"在 {folder} 中没有找到匹配 {args.pattern} 的图片")

    images, labels = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[跳过] 无法读取: {p}")
            continue
        images.append(img)
        labels.append(os.path.basename(p))

    run_calibration(images, labels, tuple(args.grid), args.square, args.out)


if __name__ == "__main__":
    main()
