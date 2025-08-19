# chocolatechip/lanes/mapper.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2
import math

from PySide6 import QtCore, QtGui, QtWidgets


def _cv_to_qpixmap(img_bgr: np.ndarray) -> QtGui.QPixmap:
    if img_bgr is None:
        return QtGui.QPixmap()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = img_rgb.strides[0]
    qimg = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def _load_tps_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r") as f:
        n = int(f.readline().strip())
        s_list, t_list = [], []
        for _ in range(n):
            sx, sy, tx, ty = map(float, f.readline().split())
            s_list.append([sx, sy])
            t_list.append([tx, ty])
    sshape = np.array(s_list, dtype=np.float32).reshape(1, -1, 2)
    tshape = np.array(t_list, dtype=np.float32).reshape(1, -1, 2)
    return sshape, tshape


class MapperWidget(QtWidgets.QWidget):
    """
    GUI to:
      - take rectified (unwarped) image, target map image, TPS pairs .out
      - compute homography prewarp + TPS
      - export legacy .map file  'x y scale 0 map_x map_y'
      - save overlays + H.npy
    """
    mapReady = QtCore.Signal(str)  # emits path to .map file

    def __init__(self, parent=None):
        super().__init__(parent)

        # Inputs
        self.rectified_path: Path | None = None
        self.map_path: Path | None = None
        self.tps_path: Path | None = None

        self.rectified_img: np.ndarray | None = None
        self.map_img: np.ndarray | None = None
        self.prewarp_img: np.ndarray | None = None
        self.tps_img: np.ndarray | None = None
        self.H: np.ndarray | None = None
        self.overlay_pre: np.ndarray | None = None
        self.overlay_tps: np.ndarray | None = None
        self.out_dir: Path | None = None
        self.last_map_file: Path | None = None

        self._last_inliers = 0
        self._last_total   = 0
        self._last_rmse    = 0.0


        # Left controls
        self.btn_rect = QtWidgets.QPushButton("Rectified…")
        self.ed_rect  = QtWidgets.QLineEdit(); self.ed_rect.setReadOnly(True)
        self.btn_map  = QtWidgets.QPushButton("Map…")
        self.ed_map   = QtWidgets.QLineEdit(); self.ed_map.setReadOnly(True)
        self.btn_tps  = QtWidgets.QPushButton("TPS .out…")
        self.ed_tps   = QtWidgets.QLineEdit(); self.ed_tps.setReadOnly(True)
        self.btn_outd = QtWidgets.QPushButton("Output dir…")
        self.ed_outd  = QtWidgets.QLineEdit(); self.ed_outd.setReadOnly(True)

        self.alpha = QtWidgets.QDoubleSpinBox(); self.alpha.setRange(0.0,1.0); self.alpha.setSingleStep(0.05); self.alpha.setValue(0.6)
        self.ransac = QtWidgets.QDoubleSpinBox(); self.ransac.setRange(0.1,100.0); self.ransac.setSingleStep(0.5); self.ransac.setValue(3.0)

        self.btn_run  = QtWidgets.QPushButton("Run & Export .map")
        self.lbl_info = QtWidgets.QLabel("Ready.")
        self.lbl_info.setWordWrap(True)

        form = QtWidgets.QFormLayout()
        form.addRow(self.btn_rect, self.ed_rect)
        form.addRow(self.btn_map,  self.ed_map)
        form.addRow(self.btn_tps,  self.ed_tps)
        form.addRow(self.btn_outd, self.ed_outd)
        form.addRow("Overlay α:", self.alpha)
        form.addRow("RANSAC thresh (px):", self.ransac)
        form_box = QtWidgets.QGroupBox("Inputs / Settings")
        v = QtWidgets.QVBoxLayout(); v.addLayout(form); v.addWidget(self.btn_run); v.addWidget(self.lbl_info)
        form_box.setLayout(v)

        # Right previews (tabs)
        self.prev_tabs = QtWidgets.QTabWidget()
        self.lbl_prewarp  = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.lbl_tps      = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.lbl_ol_pre   = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.lbl_ol_tps   = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        for lab in (self.lbl_prewarp, self.lbl_tps, self.lbl_ol_pre, self.lbl_ol_tps):
            lab.setMinimumSize(480, 320)
            lab.setStyleSheet("QLabel { background: #111; border: 1px solid #333; }")
        self.prev_tabs.addTab(self.lbl_prewarp, "Prewarp")
        self.prev_tabs.addTab(self.lbl_tps,     "TPS")
        self.prev_tabs.addTab(self.lbl_ol_pre,  "Overlay (Prewarp)")
        self.prev_tabs.addTab(self.lbl_ol_tps,  "Overlay (TPS)")

        # Layout
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(form_box, 0)
        lay.addWidget(self.prev_tabs, 1)

        # Signals
        self.btn_rect.clicked.connect(self._pick_rectified)
        self.btn_map.clicked.connect(self._pick_map)
        self.btn_tps.clicked.connect(self._pick_tps)
        self.btn_outd.clicked.connect(self._pick_outdir)
        self.btn_run.clicked.connect(self._run)

        self.setAcceptDrops(True)

    # Drag & drop
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
    def dropEvent(self, e: QtGui.QDropEvent) -> None:
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            ext = p.suffix.lower()
            if ext in (".png",".jpg",".jpeg",".bmp"):
                if "unwarped" in p.stem.lower():
                    self._set_rectified(p)
                elif "map" in p.stem.lower():
                    self._set_map(p)
            elif ext == ".out":
                self._set_tps(p)

    # File pickers
    def _pick_rectified(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Rectified image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path: self._set_rectified(Path(path))
    def _pick_map(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Map image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path: self._set_map(Path(path))
    def _pick_tps(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "TPS pairs (.out)", "", "Point Pairs (*.out)")
        if path: self._set_tps(Path(path))
    def _pick_outdir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Output directory")
        if path:
            self.out_dir = Path(path)
            self.ed_outd.setText(str(self.out_dir))

    # Setters
    def _set_rectified(self, p: Path):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            QtWidgets.QMessageBox.critical(self, "Load failed", f"Could not read: {p}"); return
        self.rectified_path, self.rectified_img = p, img
        self.ed_rect.setText(str(p))
    def _set_map(self, p: Path):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            QtWidgets.QMessageBox.critical(self, "Load failed", f"Could not read: {p}"); return
        self.map_path, self.map_img = p, img
        self.ed_map.setText(str(p))
    def _set_tps(self, p: Path):
        self.tps_path = p
        self.ed_tps.setText(str(p))

    # External seeding (optional)
    def suggest_files(self, rectified: str|Path|None, map_img: str|Path|None, tps_out: str|Path|None, out_dir: str|Path|None=None):
        if rectified: self._set_rectified(Path(rectified))
        if map_img:   self._set_map(Path(map_img))
        if tps_out:   self._set_tps(Path(tps_out))
        if out_dir:
            self.out_dir = Path(out_dir); self.ed_outd.setText(str(self.out_dir))

    def _run(self):
        try:
            self._do_process()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def _do_process(self):
        if self.rectified_img is None or self.map_img is None or self.tps_path is None:
            raise RuntimeError("Please choose rectified image, map image, and TPS .out first.")
        out_dir = self.out_dir or (self.rectified_path.parent / "output_align")
        out_dir.mkdir(parents=True, exist_ok=True)

        rect = self.rectified_img
        target = self.map_img
        sshape, tshape = _load_tps_pairs(self.tps_path)
        if sshape.shape[1] < 4:
            raise ValueError("Need at least 4 pairs for a robust homography prewarp.")

        # --- Homography prewarp (RANSAC)
        H, mask = cv2.findHomography(sshape.reshape(-1,2), tshape.reshape(-1,2),
                                    method=cv2.RANSAC, ransacReprojThreshold=float(self.ransac.value()))
        if H is None:
            raise RuntimeError("Homography estimation failed; check spread/quality of points.")
        self.H = H
        inliers = mask.ravel().astype(bool)
        s_in = sshape[:, inliers, :]
        t_in = tshape[:, inliers, :]

        proj = cv2.perspectiveTransform(s_in.astype(np.float32), H)
        res = (t_in - proj).reshape(-1, 2)
        rmse = float(np.sqrt((res**2).sum(axis=1).mean()))

        # Save for later + show interim
        self._last_inliers = int(inliers.sum())
        self._last_total   = int(len(inliers))
        self._last_rmse    = rmse
        self.lbl_info.setText(f"Homography inliers {self._last_inliers}/{self._last_total} "
                            f"({self._last_inliers / max(self._last_total,1):.0%}), RMSE={rmse:.2f}px")


        # Prewarp to map frame
        self.prewarp_img = cv2.warpPerspective(rect, H, (target.shape[1], target.shape[0]))

        # Transform inlier source points into prewarp frame
        s_pre = cv2.perspectiveTransform(s_in, H)

        # --- TPS refinement (prewarp → map)
        tps = cv2.createThinPlateSplineShapeTransformer(0)
        matches = [cv2.DMatch(i, i, 0) for i in range(s_pre.shape[1])]
        tps.estimateTransformation(t_in, s_pre, matches)
        self.tps_img = tps.warpImage(self.prewarp_img)

        # --- Direct TPS map export (rectified → map)
        tps2 = cv2.createThinPlateSplineShapeTransformer(0)
        matches_all = [cv2.DMatch(i, i, 0) for i in range(sshape.shape[1])]
        tps2.estimateTransformation(sshape, tshape, matches_all)

        Hh, Ww = target.shape[:2]
        Hr, Wr = rect.shape[:2]

        xx, yy = np.meshgrid(np.arange(Wr, dtype=np.float32),
                             np.arange(Hr, dtype=np.float32))
        pts_rect = np.stack([xx.ravel(), yy.ravel()], axis=1).reshape(1, -1, 2)

        _, pts_map = tps2.applyTransformation(pts_rect)
        pts_map = pts_map.reshape(-1, 2)

        base = np.column_stack([
            xx.ravel(), yy.ravel(),
            np.zeros(xx.size, np.float32),
            np.zeros(xx.size, np.float32),
            pts_map[:, 0], pts_map[:, 1]
        ]).astype(np.float32)

        # Sparse table Tm + LR bounds
        # Build sparse table from rect→map samples (width-major: [x, y])

        Tm = np.zeros((Wr, Hr, 2), dtype=np.float32)
        lr = np.zeros((Hr, 2), dtype=np.float32)
        base_i = np.rint(base).astype(np.int32)

        # accept if within an expanded window (old program used [-50, W+50] etc.)
        for i in range(base.shape[0]):
            mx, my = base[i, 4], base[i, 5]
            if (-50 < my < Hh + 50) and (-50 < mx < Ww + 50):
                x, y = base_i[i, 0], base_i[i, 1]
                if 0 <= x < Wr and 0 <= y < Hr:
                    Tm[x, y] = mx, my

        # Left/right per scanline (y = 0..Hr-1)
        for i in range(Hr):
            # leftmost
            for j in range(Wr):
                if np.all(Tm[j, i]):   # instead of (Tm[j,i,0] != 0 or Tm[j,i,1] != 0)
                    lr[i, 0] = j; break
            # rightmost
            for j in range(Wr - 1, -1, -1):
                if np.all(Tm[j, i]):
                    lr[i, 1] = j; break


        # Old-style monotonic smoothing of LR envelope
        plr = lr.copy()

        # Find rows that actually have samples
        rows_left  = np.where(lr[:, 0] > 0)[0]
        rows_right = np.where(lr[:, 1] > 0)[0]
        if rows_left.size == 0 or rows_right.size == 0:
            raise RuntimeError("No valid rows found to build envelope; check TPS pairs / coverage.")

        # indices delimiting the useful band (like old ll/rr)
        ll = int(rows_left[np.argmin(lr[rows_left, 0])])
        rr = int(rows_right[np.argmax(lr[rows_right, 1])])


        # Left edge should not expand outward
        for i in range(1, ll, 1):
            if lr[i-1, 0] == 0: 
                continue
            if plr[i, 0] > plr[i-1, 0]:
                plr[i, 0] = plr[i-1, 0]
        for i in range(Hr - 2, ll, -1):
            if lr[i+1, 0] == 0:
                continue
            if plr[i, 0] > plr[i+1, 0]:
                plr[i, 0] = plr[i+1, 0]

        # Right edge should not shrink inward
        for i in range(1, rr, 1):
            if lr[i-1, 1] == 0:
                continue
            if plr[i, 1] < plr[i-1, 1]:
                plr[i, 1] = plr[i-1, 1]
        for i in range(Hr - 2, rr, -1):
            if lr[i+1, 1] == 0:
                continue
            if plr[i, 1] < plr[i+1, 1]:
                plr[i, 1] = plr[i+1, 1]


        # KD interpolation (prefer SciPy; fall back to simple 4-NN via numpy if missing)
        try:
            from scipy.spatial import KDTree  # type: ignore
            use_scipy = True
        except Exception:
            use_scipy = False

        valid = []
        for i in range(base.shape[0]):
            mx, my = base[i, 4], base[i, 5]
            if -50 < my < Hh + 50 and -50 < mx < Ww + 50:
                valid.append(base[i])
        valid = np.array(valid, dtype=np.float32)
        if valid.size == 0:
            raise RuntimeError("No valid samples found to build map. Check pairs / frames.")

        ntm = Tm.copy()
        rows = []
        if use_scipy:
            kd = KDTree(valid[:, :2])
            for i in range(Hr):
                if plr[i,0] == 0 and plr[i,1] == 0:
                    continue
                bl, br = int(plr[i, 0]), int(plr[i, 1])
                if br <= bl:
                    br = min(bl + 2, Wr)  # tiny span to avoid zero-width band

                for j in range(bl, br):
                    if ntm[j, i, 0] == 0 and ntm[j, i, 1] == 0:
                        d, pos = kd.query([j, i], k=min(4, len(valid)))
                        d = np.atleast_1d(d); pos = np.atleast_1d(pos)
                        w = (1.0 / np.maximum(d, 1e-6)); w /= w.sum()
                        ntm[j, i] = (valid[pos, 4:6] * w[:,None]).sum(axis=0)
                    rows.append([j, i, 0, 0, ntm[j, i, 0], ntm[j, i, 1]])
        else:
            # crude 4-NN using numpy distance
            pts = valid[:, :2]
            for i in range(Hr):
                if plr[i,0] == 0 and plr[i,1] == 0:
                    continue
                bl, br = int(plr[i,0]), int(plr[i,1])
                grid_y = i
                for j in range(bl, br):
                    if ntm[j, i, 0] == 0 and ntm[j, i, 1] == 0:
                        q = np.array([j, grid_y], dtype=np.float32)
                        d = np.linalg.norm(pts - q, axis=1)
                        idx = np.argpartition(d, min(3, len(d)-1))[:4]
                        dd = np.maximum(d[idx], 1e-6)
                        w = (1.0 / dd); w /= w.sum()
                        ntm[j, i] = (valid[idx, 4:6] * w[:,None]).sum(axis=0)
                    rows.append([j, i, 0, 0, ntm[j, i, 0], ntm[j, i, 1]])

        Maper3 = np.array(rows, dtype=np.float32)

        # Local 'scale' (K=21) — optional
        if Maper3.shape[0] >= 22:
            try:
                if use_scipy:
                    kd2 = KDTree(Maper3[:, :2])
                    for i in range(Hr):
                        if plr[i,0] == 0 and plr[i,1] == 0: continue
                        bl, br = int(plr[i,0]), int(plr[i,1])
                        for j in range(bl, br):
                            d, idx = kd2.query([j, i], k=min(21, len(Maper3)))
                            if np.isscalar(idx): continue
                            pts = Maper3[idx[1:], 4:6] - Maper3[idx[0], 4:6]
                            s1 = np.sum(d) if np.ndim(d) else float(d)
                            s2 = float(np.sqrt((pts**2).sum(1)).sum())
                            Maper3[idx[0], 2] = (s2 / s1) if s1 != 0 else 0.0
            except Exception:
                pass  # non-fatal

        # Save outputs
        stem = (self.rectified_path.stem.replace("_unwarped", "") if self.rectified_path else "frame")
        pre_path = out_dir / f"{stem}_prewarp.png"
        tps_path = out_dir / f"{stem}_TPS.png"
        H_path   = out_dir / f"{stem}_H.npy"

        cv2.imwrite(str(pre_path), self.prewarp_img)
        cv2.imwrite(str(tps_path), self.tps_img)
        np.save(H_path, self.H)

        a = float(np.clip(self.alpha.value(), 0.0, 1.0))
        self.overlay_pre = cv2.addWeighted(self.map_img, a, self.prewarp_img, 1 - a, 0)
        self.overlay_tps = cv2.addWeighted(self.map_img, a, self.tps_img, 1 - a, 0)

        ol_pre_path = out_dir / f"{stem}_overlay_prewarp.png"
        ol_tps_path = out_dir / f"{stem}_overlay_tps.png"
        cv2.imwrite(str(ol_pre_path), self.overlay_pre)
        cv2.imwrite(str(ol_tps_path), self.overlay_tps)

        map_path = out_dir / f"{stem}_tps_b_{rect.shape[1]}.map"
        np.savetxt(map_path, np.round(Maper3, 2), fmt="%.2f %.2f %.2f %.2f %.2f %.2f")
        self.last_map_file = map_path

        # Previews
        self._set_preview(self.lbl_prewarp, self.prewarp_img)
        self._set_preview(self.lbl_tps, self.tps_img)
        self._set_preview(self.lbl_ol_pre, self.overlay_pre)
        self._set_preview(self.lbl_ol_tps, self.overlay_tps)

        self.lbl_info.setText(
            f"✔ Wrote .map: {map_path.name}\n"
            f"Inliers: {self._last_inliers}/{self._last_total} "
            f"({self._last_inliers / max(self._last_total,1):.0%}) • RMSE={self._last_rmse:.2f}px"
        )

        self.mapReady.emit(str(map_path))

    def _set_preview(self, label: QtWidgets.QLabel, img_bgr: np.ndarray | None):
        if img_bgr is None:
            label.clear(); return
        pm = _cv_to_qpixmap(img_bgr)
        label.setPixmap(pm.scaled(label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        for lab, img in ((self.lbl_prewarp, self.prewarp_img),
                         (self.lbl_tps, self.tps_img),
                         (self.lbl_ol_pre, self.overlay_pre),
                         (self.lbl_ol_tps, self.overlay_tps)):
            if img is not None:
                self._set_preview(lab, img)
