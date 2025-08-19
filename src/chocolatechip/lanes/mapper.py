# chocolatechip/lanes/mapper.py
from __future__ import annotations

import os, math, pickle, traceback
from pathlib import Path
import numpy as np
import numpy.ma as ma
import cv2
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from PySide6 import QtCore, QtGui, QtWidgets


# ---------- Legacy constants (unchanged) ----------
WIDTH  = 1280
HEIGHT = 960
F      = 355           # focal length (px)
Z_COORD = -80          # ground plane z
YAW_DEG = 125
ROLL_DEG = 0


def _cv_to_qpixmap(img_bgr: np.ndarray) -> QtGui.QPixmap:
    if img_bgr is None:
        return QtGui.QPixmap()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    qimg = QtGui.QImage(img_rgb.data, w, h, img_rgb.strides[0], QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / (math.sqrt(np.dot(axis, axis)) + 1e-12)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([
        [aa + bb - cc - dd, 2*(bc + ad),     2*(bd - ac)],
        [2*(bc - ad),       aa + cc - bb - dd, 2*(cd + ab)],
        [2*(bd + ac),       2*(cd - ab),     aa + dd - bb - cc]
    ])


def run_legacy_pipeline(raw_path: str, map_path: str, tps_out_path: str,
                        out_dir: str|None=None, log=lambda *_: None):
    """
    Runs the legacy pipeline unchanged and returns:
      out_map_path, previews_dict
    where previews_dict contains BGR images:
      {'out_image', 'tps_img', 'overlay', 'scale_img', 'boundary_img'}
    """
    # --------- Setup / I/O ----------
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_files_dir = os.path.join(out_dir, "output_files")
    else:
        out_files_dir = os.path.join(os.path.dirname(raw_path), "output_files")
        out_dir = os.path.dirname(raw_path)
    os.makedirs(out_files_dir, exist_ok=True)

    width, height = WIDTH, HEIGHT
    width_half, height_half = width // 2, height // 2

    frame = cv2.imread(raw_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read raw fisheye still: {raw_path}")
    if frame.shape[1] != width or frame.shape[0] != height:
        raise ValueError(f"RAW must be exactly {WIDTH}x{HEIGHT}. Got {frame.shape[1]}x{frame.shape[0]}.")

    Map = cv2.imread(map_path)
    if Map is None:
        raise FileNotFoundError(f"Could not read map image: {map_path}")
    if Map.shape[1] != width or Map.shape[0] != height:
        raise ValueError(f"MAP must be exactly {WIDTH}x{HEIGHT}. Got {Map.shape[1]}x{Map.shape[0]}.")

    if not os.path.isfile(tps_out_path):
        raise FileNotFoundError(f"TPS pairs file not found: {tps_out_path}")

    # --------- Build unit direction map (equisolid) ----------
    log("Creating DIR_map…")
    radius = min(width, height) // 2
    All = [[x - width_half, y - height_half] for x in range(width) for y in range(height)]

    DIR_map = []
    for a0, a1 in All:
        if a0 == 0 and a1 == 0:
            DIR_map.append([0, 0, 0, 0, -1]); continue
        r = math.hypot(a0, a1)
        if r > radius: continue
        if r / (2 * F) >= 1: continue
        theta = 2 * math.asin(r / (2 * F))
        z = [0, 0, -1]
        v2 = np.cross([a0, a1, 0], z)
        Rm = rotation_matrix(v2, math.radians(90) - theta)
        Dir = np.dot(Rm, np.asarray([a0, a1, 0]))
        Dir_n = Dir / (np.linalg.norm(Dir) + 1e-12)
        if np.isnan(Dir_n).any(): continue
        DIR_map.append([a0, a1, Dir_n[0], Dir_n[1], Dir_n[2]])

    DIR_map = np.array(DIR_map)
    X = np.float64(DIR_map[:, 2:5])
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)

    # Optional: save pickles (exactly like legacy)
    with open(os.path.join(out_files_dir, 'equisolid_DIR_map'), 'wb') as f:
        pickle.dump(DIR_map, f)
    with open(os.path.join(out_files_dir, 'equisolid_nbrs'), 'wb') as f:
        pickle.dump(nbrs, f)

    # --------- Template creation & orientation ----------
    log("Creating template…")
    grnd_vw = [[x - height_half, y - width_half, Z_COORD] for x in np.arange(0, height) for y in np.arange(0, width)]
    v2 = [0, 0, 1]
    R_matrix = rotation_matrix(v2, math.radians(YAW_DEG))
    pt2 = [np.dot(R_matrix, np.asarray(gv)) for gv in grnd_vw]
    v3 = np.dot(R_matrix, np.asarray([0, 1, 0]))
    R_matrix = rotation_matrix(v3, math.radians(ROLL_DEG))
    pt3 = [list(np.dot(R_matrix, np.asarray(gv))) for gv in pt2]
    pt3 = np.array(pt3)
    Template_u = [p / (np.linalg.norm(p) + 1e-12) for p in pt3]

    # --------- NN mapping ----------
    log("Mapping fisheye → map…")
    _, indices = nbrs.kneighbors(Template_u)

    grnd_vw_out = [[x, y] for x in np.arange(0, height) for y in np.arange(0, width)]
    A = np.array(grnd_vw_out)
    B = np.add(
        DIR_map[:, 0:2][indices].reshape(len(indices), 8),
        [width_half, height_half, width_half, height_half, width_half, height_half, width_half, height_half]
    )
    Maper = np.concatenate((A, B), 1)

    # --------- Create unwarped composite (preview parity) ----------
    log("Compositing unwarped preview…")
    Out_image = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(len(indices)):
        r_out = int(Maper[i][0]); c_out = int(Maper[i][1])
        x0, y0, x1, y1, x2, y2, x3, y3 = Maper[i][2:10]
        x0 = np.clip(int(x0), 0, width - 1);  y0 = np.clip(int(y0), 0, height - 1)
        x1 = np.clip(int(x1), 0, width - 1);  y1 = np.clip(int(y1), 0, height - 1)
        x2 = np.clip(int(x2), 0, width - 1);  y2 = np.clip(int(y2), 0, height - 1)
        x3 = np.clip(int(x3), 0, width - 1);  y3 = np.clip(int(y3), 0, height - 1)
        Out_image[r_out, c_out] = np.mean([
            frame[y0, x0], frame[y1, x1], frame[y2, x2], frame[y3, x3]
        ], axis=0)

    # --------- TPS warp using .out ----------
    log(f"Reading TPS pairs: {tps_out_path}")
    tps = cv2.createThinPlateSplineShapeTransformer(0)
    tps2 = cv2.createThinPlateSplineShapeTransformer(0)

    sshape, tshape, matches = [], [], []
    with open(tps_out_path, 'r') as pfile:
        n = int(pfile.readline())
        for i in range(n):
            point = np.float32(pfile.readline().split(' '))
            sshape.append(point[0:2])
            tshape.append(point[2:4])
            matches.append(cv2.DMatch(i, i, 0))
    sshape = np.array(sshape).reshape(1, -1, 2)
    tshape = np.array(tshape).reshape(1, -1, 2)

    # Legacy directions:
    tps.estimateTransformation(tshape, sshape, matches)
    tps2.estimateTransformation(sshape, tshape, matches)

    TPS_img = tps.warpImage(Out_image)
    # --- TPS warp using .out (already computed) ---
    TPS_img = tps.warpImage(Out_image)

    # Convert to uint8 before blending (Map is uint8 from imread)
    TPS_u8 = np.uint8(np.clip(TPS_img, 0, 255))
    overlay = cv2.addWeighted(Map, 0.6, TPS_u8, 0.4, 0)


    # --------- Dense mapping via TPS (legacy) ----------
    log("Building dense TPS map…")
    Maper = Maper[:, :4]
    pts = np.fliplr(Maper[:, :2])  # (x, y)
    Maper1 = np.concatenate((Maper[:, 2:4], pts), 1)
    pts_f32 = pts.astype(np.float32).reshape(1, -1, 2)
    _, output = tps2.applyTransformation(pts_f32)
    output = output.reshape(Maper1.shape[0], -1)
    Maper1 = np.concatenate((Maper1, output), 1)

    Tm = np.zeros((width, height, 2))
    lr = np.zeros((height, 2))
    Maper2 = np.rint(Maper1).astype(np.int32)

    for i in range(len(pts)):
        y_out = Maper2[i][0]
        x_out = Maper2[i][1]
        tx = Maper1[i][4]; ty = Maper1[i][5]
        if -50 < ty < height + 50 and -50 < tx < width + 50:
            Tm[y_out, x_out] = tx, ty

    for i in range(height):
        for j in range(width):
            if np.all(Tm[j, i]): lr[i, 0] = j; break
        for j in range(width - 1, -1, -1):
            if np.all(Tm[j, i]): lr[i, 1] = j; break

    ll = int(np.argmin(ma.masked_where(lr[:, 0] == 0, lr[:, 0])))
    rr = int(np.argmax(lr[:, 1]))
    plr = np.copy(lr)

    for i in range(1, ll, 1):
        if lr[i - 1, 0] == 0: continue
        if plr[i, 0] > plr[i - 1, 0]: plr[i, 0] = plr[i - 1, 0]
    for i in range(height - 2, ll, -1):
        if lr[i + 1, 0] == 0: continue
        if plr[i, 0] > plr[i + 1, 0]: plr[i, 0] = plr[i + 1, 0]
    for i in range(1, rr, 1):
        if lr[i - 1, 1] == 0: continue
        if plr[i, 1] < plr[i - 1, 1]: plr[i, 1] = plr[i - 1, 1]
    for i in range(height - 2, rr, -1):
        if lr[i + 1, 1] == 0: continue
        if plr[i, 1] < plr[i + 1, 1]: plr[i, 1] = plr[i + 1, 1]

    pts_valid = []
    for i in range(len(pts)):
        if -50 < Maper1[i][5] < height + 50 and -50 < Maper1[i][4] < width + 50:
            pts_valid.append(Maper1[i])
    pts_valid = np.array(pts_valid)
    kd = KDTree(np.array(pts_valid[:, :2]))

    log("Interpolating field…")
    ntm = Tm.copy()
    rows = []
    for i in range(height):
        if np.all(lr[i]):
            bl, br = int(plr[i, 0]), int(plr[i, 1])
            for j in range(bl, br):
                if ntm[j, i, 0] == 0 and ntm[j, i, 1] == 0:
                    d, pos = kd.query([j, i], k=4)
                    w = 1 / np.maximum(d, 1e-12)
                    w = w / np.sum(w)
                    for k in range(4):
                        ntm[j, i] = ntm[j, i] + pts_valid[pos, 4:6][k] * w[k]
                rows.append([j, i, 0, 0, ntm[j, i, 0], ntm[j, i, 1]])

    Maper3 = np.array(rows)
    kd2 = KDTree(Maper3[:, :2])

    log("Estimating local scale…")
    for i in range(height):
        if np.all(lr[i]):
            bl, br = int(plr[i, 0]), int(plr[i, 1])
            for j in range(bl, br):
                d, args = kd2.query([j, i], k=min(21, len(Maper3)))
                neigh = Maper3[args[1:], 4:6] - Maper3[args[0], 4:6]
                sumd1 = d.sum() if np.ndim(d) else float(d)
                sumd2 = (np.sqrt((neigh ** 2).sum(1))).sum() if len(neigh) else 0.0
                scale = (sumd2 / sumd1) if sumd1 else 0.0
                Maper3[args[0], 2] = scale

    # Visualize scale (grayscale heat)
    scale_img = np.zeros((height, width, 3), np.uint8)
    if Maper3.shape[0] > 0 and np.max(Maper3[:, 2]) > 0:
        fac = 255.0 / max(1e-6, Maper3[:, 2].max())
        for i in range(Maper3.shape[0]):
            r = int(Maper3[i, 1]); c = int(Maper3[i, 0])
            if 0 <= r < height and 0 <= c < width:
                val = int(fac * Maper3[i, 2])
                scale_img[r, c] = (val, val, val)

    # Boundary polygon preview
    polygon = []
    for i in range(height):
        if np.all(lr[i]):
            bl, br = plr[i]; polygon.append([int(bl), i])
    for i in range(height - 1, -1, -1):
        if np.all(lr[i]):
            bl, br = plr[i]; polygon.append([int(br), i])
    polygon = np.array(polygon, dtype=np.int32)
    boundary_img = np.zeros((height, width, 3), np.uint8)
    if polygon.size:
        boundary_img = cv2.polylines(boundary_img, [polygon], True, (255, 255, 255), 3)

    # --------- Save final .map exactly like legacy ----------
    Maper3 = np.around(Maper3, 2)
    out_map_path = os.path.join(out_dir, "finalmap.map")
    with open(out_map_path, "w") as pts_map:
        for i in range(Maper3.shape[0]):
            print(Maper3[i][0], Maper3[i][1], Maper3[i][2], Maper3[i][3], Maper3[i][4], Maper3[i][5], file=pts_map)

    log(f"✔ Done. Wrote {out_map_path}")

    previews = {
        "out_image": np.uint8(np.clip(Out_image, 0, 255)),
        "tps_img":   TPS_u8,                 # use the uint8 version
        "overlay":   np.uint8(np.clip(overlay, 0, 255)),
        "scale_img": scale_img,
        "boundary_img": boundary_img
    }

    return out_map_path, previews


class MapperWidget(QtWidgets.QWidget):
    """
    Minimal GUI wrapper around the legacy pipeline with visual sanity previews.
    """
    mapReady = QtCore.Signal(str)  # emits path to .map file

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LegacyTPSMapper")

        # State
        self.raw_path: str|None = None
        self.map_path: str|None = None
        self.tps_path: str|None = None
        self.out_dir:  str|None = None
        self.last_map_file: str|None = None
        self._previews: dict[str, np.ndarray] | None = None

        # Inputs
        self.ed_raw = QtWidgets.QLineEdit(); self.ed_raw.setReadOnly(True)
        self.ed_map = QtWidgets.QLineEdit(); self.ed_map.setReadOnly(True)
        self.ed_out = QtWidgets.QLineEdit(); self.ed_out.setReadOnly(True)
        self.ed_dir = QtWidgets.QLineEdit(); self.ed_dir.setReadOnly(True)

        b_raw = QtWidgets.QPushButton("Raw fisheye…")
        b_map = QtWidgets.QPushButton("Google Map…")
        b_out = QtWidgets.QPushButton("TPS .out…")
        b_dir = QtWidgets.QPushButton("Output dir…")
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_run.setStyleSheet("font-weight:600;")

        # Preview widgets
        self.alpha = QtWidgets.QDoubleSpinBox(); self.alpha.setRange(0.0,1.0); self.alpha.setSingleStep(0.05); self.alpha.setValue(0.6)
        self.btn_save_prev = QtWidgets.QPushButton("Save previews")
        self.btn_save_prev.setEnabled(False)

        self.tabs = QtWidgets.QTabWidget()
        self.lbl_out   = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.lbl_tps   = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.lbl_ol    = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.lbl_scale = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.lbl_poly  = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        for lab in (self.lbl_out, self.lbl_tps, self.lbl_ol, self.lbl_scale, self.lbl_poly):
            lab.setMinimumSize(480, 320)
            lab.setStyleSheet("QLabel { background:#111; border:1px solid #333; }")

        self.tabs.addTab(self.lbl_out,   "Unwarped")
        self.tabs.addTab(self.lbl_tps,   "TPS result")
        self.tabs.addTab(self.lbl_ol,    "Overlay")
        self.tabs.addTab(self.lbl_scale, "Scale heat")
        self.tabs.addTab(self.lbl_poly,  "Boundary")

        # Log
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setStyleSheet("QPlainTextEdit { background:#111; color:#ddd; }")

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow(b_raw, self.ed_raw)
        form.addRow(b_map, self.ed_map)
        form.addRow(b_out, self.ed_out)
        form.addRow(b_dir, self.ed_dir)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Overlay α:"))
        h.addWidget(self.alpha)
        h.addStretch(1)
        h.addWidget(self.btn_save_prev)

        left = QtWidgets.QVBoxLayout()
        left.addLayout(form)
        left.addWidget(self.btn_run)
        left.addLayout(h)
        left.addWidget(self.log, 1)

        lay = QtWidgets.QHBoxLayout(self)
        box = QtWidgets.QGroupBox("Inputs / Actions")
        box.setLayout(left)
        lay.addWidget(box, 0)
        lay.addWidget(self.tabs, 1)

        # Signals
        b_raw.clicked.connect(self._pick_raw)
        b_map.clicked.connect(self._pick_map)
        b_out.clicked.connect(self._pick_out)
        b_dir.clicked.connect(self._pick_dir)
        self.btn_run.clicked.connect(self._run)
        self.btn_save_prev.clicked.connect(self._save_previews)
        self.alpha.valueChanged.connect(self._refresh_overlay)

        # Drag & drop
        self.setAcceptDrops(True)

    # Public helper (optional)
    def suggest_files(self, raw: str|Path|None, map_img: str|Path|None, tps_out: str|Path|None, out_dir: str|Path|None=None):
        if raw:     self._set_raw(str(raw))
        if map_img: self._set_map(str(map_img))
        if tps_out: self._set_out(str(tps_out))
        if out_dir: self._set_dir(str(out_dir))

    # DnD
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent) -> None:
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e: QtGui.QDropEvent) -> None:
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            ext = Path(p).suffix.lower()
            if ext in (".png",".jpg",".jpeg",".bmp"):
                # heuristic: "map" in name = map; else assume raw
                if "map" in Path(p).stem.lower():
                    self._set_map(p)
                else:
                    self._set_raw(p)
            elif ext == ".out":
                self._set_out(p)

    # Pickers
    def _pick_raw(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Raw fisheye (1280x960)", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if p: self._set_raw(p)
    def _pick_map(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Google Map (1280x960)", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if p: self._set_map(p)
    def _pick_out(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "TPS pairs (.out)", "", "Point Pairs (*.out)")
        if p: self._set_out(p)
    def _pick_dir(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output directory")
        if p: self._set_dir(p)

    # Setters
    def _set_raw(self, p: str):
        self.raw_path = p; self.ed_raw.setText(p)
    def _set_map(self, p: str):
        self.map_path = p; self.ed_map.setText(p)
    def _set_out(self, p: str):
        self.tps_path = p; self.ed_out.setText(p)
    def _set_dir(self, p: str):
        self.out_dir = p; self.ed_dir.setText(p)

    def _append_log(self, msg: str):
        self.log.appendPlainText(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 1)

    def _run(self):
        if not (self.raw_path and self.map_path and self.tps_path):
            QtWidgets.QMessageBox.information(self, "Missing input", "Please choose raw fisheye, map image, and TPS .out.")
            return
        self.btn_run.setEnabled(False)
        try:
            out_map, previews = run_legacy_pipeline(self.raw_path, self.map_path, self.tps_path,
                                                    self.out_dir, log=self._append_log)
            self.last_map_file = out_map
            self._previews = previews
            self.mapReady.emit(out_map)

            # Show previews
            self._set_preview(self.lbl_out,   previews.get("out_image"))
            self._set_preview(self.lbl_tps,   previews.get("tps_img"))
            self._set_preview(self.lbl_scale, previews.get("scale_img"))
            self._set_preview(self.lbl_poly,  previews.get("boundary_img"))
            self._refresh_overlay()
            self.btn_save_prev.setEnabled(True)

            QtWidgets.QMessageBox.information(self, "Done", f"Wrote:\n{out_map}")
        except Exception as e:
            tb = traceback.format_exc()
            self._append_log(tb)
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
        finally:
            self.btn_run.setEnabled(True)

    def _refresh_overlay(self):
        if not self._previews: return
        tps = self._previews.get("tps_img")
        if tps is None: return
        Map = cv2.imread(self.map_path) if self.map_path else None
        if Map is None: return
        a = float(np.clip(self.alpha.value(), 0.0, 1.0))
        overlay = cv2.addWeighted(Map, a, tps, 1 - a, 0)
        self._set_preview(self.lbl_ol, overlay)
        self._previews["overlay"] = overlay  # cache for saving

    def _set_preview(self, label: QtWidgets.QLabel, img_bgr: np.ndarray | None):
        if img_bgr is None:
            label.clear(); return
        pm = _cv_to_qpixmap(img_bgr)
        label.setPixmap(pm.scaled(label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        # Rescale on resize
        for lab, key in ((self.lbl_out,"out_image"), (self.lbl_tps,"tps_img"),
                         (self.lbl_ol,"overlay"), (self.lbl_scale,"scale_img"),
                         (self.lbl_poly,"boundary_img")):
            if self._previews and self._previews.get(key) is not None:
                self._set_preview(lab, self._previews[key])

    def _save_previews(self):
        if not self._previews:
            return
        base_dir = self.out_dir or (self.raw_path and os.path.dirname(self.raw_path)) or "."
        stem = Path(self.raw_path).stem if self.raw_path else "frame"
        outs = {
            "unwarped.png": self._previews.get("out_image"),
            "tps.png":      self._previews.get("tps_img"),
            "overlay.png":  self._previews.get("overlay"),
            "scale.png":    self._previews.get("scale_img"),
            "boundary.png": self._previews.get("boundary_img"),
        }
        ok_all = True
        for name, img in outs.items():
            if img is None: continue
            p = os.path.join(base_dir, name)
            ok = cv2.imwrite(p, img)
            ok_all = ok_all and ok
        if ok_all:
            QtWidgets.QMessageBox.information(self, "Saved", f"Previews saved to:\n{base_dir}")
        else:
            QtWidgets.QMessageBox.warning(self, "Partial save", f"Some previews could not be saved to:\n{base_dir}")
