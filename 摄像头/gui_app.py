# -*- coding: utf-8 -*-
"""
相机标定与 PnP 测距 —— 桌面图形界面（tkinter）
=================================================
- 完全离线运行：不依赖服务器、浏览器或网络，图片只在本机处理
- 算法全部复用 camera_calibration.py（张正友标定 + PnP 测距）
- 依赖: numpy, opencv-python, pillow（Python 标准库自带 tkinter）

用法:
    python gui_app.py
    或双击「启动程序.bat」

三个页签:
  ① 标定    —— 添加 15~25 张棋盘格照片（或摄像头采集），一键标定，
               显示内参矩阵 / 畸变系数 / 逐张与平均重投影误差 / 去畸变对比图
  ② 测距    —— 选择含 A4 纸照片，自动检测角点或手动点 4 角，求解距离
  ③ 去畸变  —— 对任意照片去畸变并保存
"""

import glob
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import camera_calibration as cc

OUT_DIR = "calib_out"
MONO_FONT = ("Consolas", 10)


def pil_from_bgr(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class ImageCanvas(tk.Canvas):
    """显示图像并支持按顺序点选 4 个角点的画布。"""

    def __init__(self, master, max_w=860, max_h=460, clickable=True, **kw):
        super().__init__(master, bg="#101010", highlightthickness=1,
                         highlightbackground="#3a3a3a", **kw)
        self._photo = None
        self.max_w, self.max_h = max_w, max_h
        self.clickable = clickable
        self.orig_w = self.orig_h = 0
        self.disp_w = self.disp_h = 0
        self.points = []
        self.on_point = None
        self.bind("<Button-1>", self._on_click)

    def show(self, img_bgr, clear_points=True):
        """显示图像（自动缩放），points 为原图坐标。"""
        if clear_points:
            self.points = []
        h, w = img_bgr.shape[:2]
        scale = min(1.0, self.max_w / w, self.max_h / h)
        dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
        pil = pil_from_bgr(img_bgr)
        if scale < 1.0:
            pil = pil.resize((dw, dh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)
        self.delete("all")
        self.create_image(0, 0, anchor="nw", image=self._photo)
        self.config(width=dw, height=dh)
        self.orig_w, self.orig_h = w, h
        self.disp_w, self.disp_h = dw, dh
        self._redraw_points()

    def _on_click(self, event):
        if not self.clickable or not self.disp_w or len(self.points) >= 4:
            return
        x = event.x * self.orig_w / self.disp_w
        y = event.y * self.orig_h / self.disp_h
        self.points.append((x, y))
        self._redraw_points()
        if self.on_point:
            self.on_point(self.points)

    def _redraw_points(self):
        for i, (x, y) in enumerate(self.points):
            dx = x * self.disp_w / self.orig_w
            dy = y * self.disp_h / self.orig_h
            self.create_oval(dx - 7, dy - 7, dx + 7, dy + 7,
                             outline="#ff5252", width=2)
            self.create_text(dx + 12, dy + 6, text=str(i + 1),
                             fill="#ff5252", font=("Arial", 12, "bold"), anchor="nw")

    def get_points(self):
        if len(self.points) != 4:
            return None
        return np.array(self.points, dtype=np.float32)


class App:
    def __init__(self, root):
        self.root = root
        root.title("相机标定与 PnP 测距")
        root.geometry("1120x800")
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.calib_items = []          # [(路径, BGR 图)]
        self.measure_img = None        # 原始测量图
        self.measure_undist = None     # 去畸变后的测量图（显示/手动点选用）
        self.measure_K_use = None      # 去畸变后的内参
        self.measure_K = None
        self.measure_dist = None
        self.und_img = None
        self.und_result = None

        self._build_calib()
        self._build_measure()
        self._build_undistort()

    # ============================== 标定页 ==============================
    def _build_calib(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  ① 标定  ")
        top = ttk.Frame(tab)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="内角点列:").pack(side="left")
        self.v_cols = tk.IntVar(value=9)
        ttk.Entry(top, textvariable=self.v_cols, width=6).pack(side="left", padx=(2, 12))
        ttk.Label(top, text="行:").pack(side="left")
        self.v_rows = tk.IntVar(value=6)
        ttk.Entry(top, textvariable=self.v_rows, width=6).pack(side="left", padx=(2, 12))
        ttk.Label(top, text="单格边长(mm):").pack(side="left")
        self.v_square = tk.DoubleVar(value=25.0)
        ttk.Entry(top, textvariable=self.v_square, width=8).pack(side="left", padx=(2, 20))
        ttk.Button(top, text="添加照片…", command=self.add_photos).pack(side="left", padx=4)
        ttk.Button(top, text="摄像头采集…", command=self.capture_photos).pack(side="left", padx=4)
        ttk.Button(top, text="清空", command=self.clear_photos).pack(side="left", padx=4)
        ttk.Button(top, text="开始标定", command=self.run_calib).pack(side="left", padx=(16, 4))

        self.list_calib = tk.Listbox(tab, height=7)
        self.list_calib.pack(fill="x", padx=10, pady=4)
        self.lbl_count = ttk.Label(tab, text="已添加 0 张（建议 15~25 张，不同位置/角度/距离）")
        self.lbl_count.pack(anchor="w", padx=10)

        self.txt_calib = tk.Text(tab, height=16, font=MONO_FONT,
                                 bg="#0d1117", fg="#d0d7de", wrap="none")
        self.txt_calib.pack(fill="both", expand=True, padx=10, pady=6)
        btnrow = ttk.Frame(tab)
        btnrow.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Button(btnrow, text="查看去畸变对比图", command=self.show_comparison).pack(side="left")
        self.lbl_cmp = ttk.Label(btnrow, text="")
        self.lbl_cmp.pack(side="left", padx=8)

    def add_photos(self):
        paths = filedialog.askopenfilenames(
            title="选择棋盘格照片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")])
        n = 0
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                self.calib_items.append((p, img))
                n += 1
        self._refresh_calib_list()
        if n:
            messagebox.showinfo("已添加", f"本次添加 {n} 张，共 {len(self.calib_items)} 张")

    def capture_photos(self):
        n = simpledialog.askinteger(
            "摄像头采集", "拍摄数量（建议 15~25）:",
            initialvalue=20, minvalue=3, maxvalue=60, parent=self.root)
        if not n:
            return
        try:
            folder = cc.capture_images(n, 0, "captured")
        except Exception as e:
            messagebox.showerror("摄像头错误", str(e))
            return
        for p in sorted(glob.glob(os.path.join(folder, "*.png"))):
            img = cv2.imread(p)
            if img is not None:
                self.calib_items.append((p, img))
        self._refresh_calib_list()

    def clear_photos(self):
        self.calib_items = []
        self._refresh_calib_list()

    def _refresh_calib_list(self):
        self.list_calib.delete(0, "end")
        for p, _ in self.calib_items:
            self.list_calib.insert("end", os.path.basename(p))
        self.lbl_count.config(text=f"已添加 {len(self.calib_items)} 张（建议 15~25 张，不同位置/角度/距离）")

    def do_calibration(self, images, labels, grid, square):
        """核心标定逻辑（与 UI 分离，便于测试）。返回 (K, dist, mean_err, per_img, good)。"""
        return cc.run_calibration(images, labels, grid, square,
                                  out_dir=OUT_DIR, return_details=True)

    def run_calib(self):
        if len(self.calib_items) < 3:
            messagebox.showwarning("提示", "图片不足：请先添加至少 3 张棋盘格照片（建议 15~25 张）")
            return
        try:
            grid = (int(self.v_cols.get()), int(self.v_rows.get()))
            square = float(self.v_square.get())
        except Exception:
            messagebox.showerror("参数错误", "请检查内角点数量与单格边长")
            return
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            images = [img for _, img in self.calib_items]
            labels = [os.path.basename(p) for p, _ in self.calib_items]
            K, dist, mean_err, per_img, good = self.do_calibration(images, labels, grid, square)
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("标定失败", str(e))
            return
        self.root.config(cursor="")

        t = self.txt_calib
        t.delete("1.0", "end")
        t.insert("end", f"有效图片: {len(good)} / {len(images)} 张\n")
        t.insert("end", f"平均重投影误差: {mean_err:.4f} px\n\n")
        t.insert("end", "内参矩阵 K:\n")
        t.insert("end", np.array2string(K, precision=4, suppress_small=True) + "\n\n")
        t.insert("end", f"焦距: fx={K[0, 0]:.2f}  fy={K[1, 1]:.2f}\n")
        t.insert("end", f"主点: cx={K[0, 2]:.2f}  cy={K[1, 2]:.2f}\n")
        t.insert("end", f"畸变系数 [k1,k2,p1,p2,k3]: {dist.ravel()}\n\n")
        t.insert("end", "逐张重投影误差 (px):\n")
        for name, e in zip(good, per_img):
            t.insert("end", f"  {name}: {e:.4f}\n")
        self.lbl_cmp.config(text=f"对比图已保存: {OUT_DIR}/undistort_comparison.jpg")
        messagebox.showinfo("标定完成",
                            f"平均重投影误差 {mean_err:.4f} px\n"
                            f"内参已保存到 {OUT_DIR}/intrinsics.json 与 intrinsics.npz")

    def show_comparison(self):
        p = os.path.join(OUT_DIR, "undistort_comparison.jpg")
        if not os.path.isfile(p):
            messagebox.showwarning("提示", "还没有对比图，请先完成标定")
            return
        img = cv2.imread(p)
        h, w = img.shape[:2]
        scale = min(1.0, 1100 / w, 640 / h)
        pil = pil_from_bgr(img)
        if scale < 1.0:
            pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        top = tk.Toplevel(self.root)
        top.title("去畸变前后对比（左:原图 右:去畸变）")
        photo = ImageTk.PhotoImage(pil)
        lbl = ttk.Label(top, image=photo)
        lbl.image = photo
        lbl.pack(padx=8, pady=8)

    # ============================== 测距页 ==============================
    def _build_measure(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  ② 测距  ")
        top = ttk.Frame(tab)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Button(top, text="选择测量图片…", command=self.choose_measure).pack(side="left")
        ttk.Label(top, text="宽(mm):").pack(side="left", padx=(16, 2))
        self.v_pw = tk.DoubleVar(value=210.0)
        ttk.Entry(top, textvariable=self.v_pw, width=7).pack(side="left", padx=2)
        ttk.Label(top, text="高(mm):").pack(side="left", padx=(8, 2))
        self.v_ph = tk.DoubleVar(value=297.0)
        ttk.Entry(top, textvariable=self.v_ph, width=7).pack(side="left", padx=2)
        self.v_mode = tk.StringVar(value="auto")
        ttk.Radiobutton(top, text="自动检测", variable=self.v_mode, value="auto").pack(side="left", padx=(16, 2))
        ttk.Radiobutton(top, text="手动点 4 角", variable=self.v_mode, value="manual").pack(side="left", padx=2)
        ttk.Button(top, text="重置角点", command=self.reset_points).pack(side="left", padx=(8, 4))
        ttk.Button(top, text="开始测量", command=self.run_measure).pack(side="left", padx=(16, 4))
        self.lbl_measure = ttk.Label(top, text="")
        self.lbl_measure.pack(side="left", padx=8)

        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True, padx=10, pady=4)
        left = ttk.Frame(body)
        left.pack(side="left", anchor="n")
        ttk.Label(left, text="测量图（手动模式请按 左上→右上→右下→左下 点 4 角）").pack(anchor="w")
        self.canvas_measure = ImageCanvas(left, max_w=620, max_h=430)
        self.canvas_measure.on_point = lambda pts: self.lbl_measure.config(
            text=f"已点 {len(pts)}/4 个角点")
        self.canvas_measure.pack(pady=4)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        ttk.Label(right, text="测量结果").pack(anchor="w")
        self.txt_measure = tk.Text(right, height=12, font=MONO_FONT,
                                   bg="#0d1117", fg="#d0d7de", wrap="none")
        self.txt_measure.pack(fill="x", pady=(0, 8))
        ttk.Label(right, text="标注图（绿框=纸张 红点=角点 坐标轴100mm: X红/Y绿/Z蓝）").pack(anchor="w")
        self.canvas_result = ImageCanvas(right, max_w=420, max_h=330, clickable=False)
        self.canvas_result.pack(anchor="w", pady=4)

    def choose_measure(self):
        p = filedialog.askopenfilename(
            title="选择含 A4 纸的照片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")])
        if not p:
            return
        img = cv2.imread(p)
        if img is None:
            messagebox.showerror("错误", "图片读取失败")
            return
        try:
            K, dist = cc.load_intrinsics(OUT_DIR)
        except RuntimeError as e:
            messagebox.showerror("尚未标定", str(e) + "\n请先到「① 标定」页完成标定")
            return
        self.measure_img = img
        self.measure_K, self.measure_dist = K, dist
        h, w = img.shape[:2]
        if dist is not None and np.any(dist):
            newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
            self.measure_undist = cv2.undistort(img, K, dist, None, newK)
            self.measure_K_use = newK
        else:
            self.measure_undist = img
            self.measure_K_use = K
        self.canvas_measure.show(self.measure_undist)
        self.canvas_result.delete("all")
        self.txt_measure.delete("1.0", "end")
        self.lbl_measure.config(text="已就绪：选择模式后点击「开始测量」")

    def reset_points(self):
        if self.measure_undist is not None:
            self.canvas_measure.show(self.measure_undist)
            self.lbl_measure.config(text="已重置，请重新点选 4 个角点")

    def do_measure_auto(self, img, K, dist, paper):
        """自动检测测距（与 UI 分离）。失败抛 RuntimeError。"""
        return cc.measure(img, K, dist, paper, manual=False,
                          allow_manual=False, quiet=True)

    def run_measure(self):
        if self.measure_img is None:
            messagebox.showwarning("提示", "请先选择测量图片")
            return
        try:
            paper = (float(self.v_pw.get()), float(self.v_ph.get()))
        except Exception:
            messagebox.showerror("参数错误", "请检查目标宽高")
            return
        try:
            if self.v_mode.get() == "manual":
                pts = self.canvas_measure.get_points()
                if pts is None:
                    messagebox.showwarning("提示", "请先在图上按顺序点 4 个角点（左上→右上→右下→左下）")
                    return
                res = cc.solve_pnp_distance(pts, self.measure_K_use, paper)
                ann = cc.annotate_pnp(self.measure_undist, pts, self.measure_K_use, res)
            else:
                res = self.do_measure_auto(self.measure_img, self.measure_K,
                                           self.measure_dist, paper)
                ann = res["annotated"]
        except RuntimeError as e:
            messagebox.showwarning("自动检测失败", f"{e}\n请切换为「手动点 4 角」模式")
            return
        except Exception as e:
            messagebox.showerror("测量失败", str(e))
            return

        self.canvas_result.show(ann)
        t = self.txt_measure
        t.delete("1.0", "end")
        t.insert("end", f"目标尺寸: {paper[0]:.0f} x {paper[1]:.0f} mm\n")
        t.insert("end", f"相机到纸张中心距离: {res['distance_mm']:.1f} mm\n")
        t.insert("end", f"                 = {res['distance_mm'] / 1000:.3f} m\n")
        t.insert("end", f"深度 tz: {res['depth_mm']:.1f} mm\n")
        t.insert("end", f"平移向量 tvec (mm): {res['tvec'].ravel()}\n")
        t.insert("end", f"旋转向量 rvec: {res['rvec'].ravel()}\n")
        t.insert("end", f"角点重投影误差: {res['reproj_err_px']:.3f} px\n")
        self.lbl_measure.config(text=f"距离 {res['distance_mm']:.0f} mm")
        messagebox.showinfo("测量完成",
                            f"相机到纸张中心距离:\n{res['distance_mm']:.1f} mm = {res['distance_mm'] / 1000:.3f} m")

    # ============================== 去畸变页 ==============================
    def _build_undistort(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  ③ 去畸变  ")
        top = ttk.Frame(tab)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Button(top, text="选择图片…", command=self.choose_und).pack(side="left")
        ttk.Button(top, text="去畸变", command=self.run_und).pack(side="left", padx=8)
        ttk.Button(top, text="保存结果…", command=self.save_und).pack(side="left")
        self.lbl_und = ttk.Label(top, text="")
        self.lbl_und.pack(side="left", padx=8)
        self.canvas_und = ImageCanvas(tab, clickable=False)
        self.canvas_und.pack(fill="both", expand=True, padx=10, pady=8)

    def choose_und(self):
        p = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")])
        if not p:
            return
        img = cv2.imread(p)
        if img is None:
            messagebox.showerror("错误", "图片读取失败")
            return
        self.und_img = img
        self.und_result = None
        self.canvas_und.show(img)
        self.lbl_und.config(text="已载入，点击「去畸变」")

    def run_und(self):
        if self.und_img is None:
            messagebox.showwarning("提示", "请先选择图片")
            return
        try:
            K, dist = cc.load_intrinsics(OUT_DIR)
        except RuntimeError as e:
            messagebox.showerror("尚未标定", str(e) + "\n请先到「① 标定」页完成标定")
            return
        h, w = self.und_img.shape[:2]
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(self.und_img, K, dist, None, newK)
        self.und_result = dst
        self.canvas_und.show(dst)
        self.lbl_und.config(text="去畸变完成，可「保存结果」")

    def save_und(self):
        if self.und_result is None:
            messagebox.showwarning("提示", "还没有去畸变结果")
            return
        p = filedialog.asksaveasfilename(
            title="保存去畸变图", defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if p:
            cv2.imwrite(p, self.und_result)
            messagebox.showinfo("已保存", f"已保存到 {p}")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
