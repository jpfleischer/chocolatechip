"""
Point Pairer — Manual control-point collector (with 90° rotate, .out export, and pipeline run)
--------------------------------------------------------------------------------------------
- Unwarped and Google Maps MUST be 1280x960 (enforced on open).
- Click Unwarped, then Google Maps to create a pair. Color-coded across panes & table.
- Export legacy .out format: first line N, then "unwarped_x unwarped_y map_x map_y".

Run standalone:
  python -m chocolatechip.lanes.pairer
"""
from __future__ import annotations

import os
import json
from typing import List, Optional, Tuple

from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.transforms import Affine2D
import numpy as np

import re

REQUIRED_W, REQUIRED_H = 1280, 960



def parse_legacy_out_pairs(path: str):
    """
    Parse legacy .out mapping files:
      - First non-empty, non-comment line is an integer N (pair count).
      - Followed by N lines of 4 floats: unwarped_x unwarped_y map_x map_y
      - Allows tabs / extra spaces, blank lines, inline or full-line comments (# or //)
    Returns: [((rx, ry), (mx, my)), ...]
    Raises: ValueError on unrecoverable parse problems.
    """
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Strip comments and blank lines
    def _clean(line: str) -> str:
        # remove inline comments
        line = re.split(r"(#|//)", line, maxsplit=1)[0]
        return line.strip()

    cleaned = [ _clean(ln) for ln in lines ]
    cleaned = [ ln for ln in cleaned if ln ]  # drop empties

    if not cleaned:
        raise ValueError("File is empty after removing comments/blank lines.")

    # First line = N
    try:
        n_expected = int(float(cleaned[0]))
    except Exception:
        raise ValueError(f"First data line must be pair count (int). Got: {cleaned[0]!r}")

    rows = cleaned[1:]
    # Parse rows that look like 4 floats
    for i, ln in enumerate(rows, start=1):
        parts = re.split(r"[,\s]+", ln.strip())
        if len(parts) < 4:
            # tolerate extra lines that aren't data; skip quietly
            continue
        try:
            rx, ry, mx, my = map(float, parts[:4])
        except Exception:
            # not a data row; skip
            continue
        pairs.append(((rx, ry), (mx, my)))

    if n_expected != len(pairs):
        # Don't hard-fail; warn via exception message the caller can surface or ignore
        # (We’ll show a QMessageBox warning in the UI layer.)
        return pairs, f"Count mismatch: header said {n_expected}, parsed {len(pairs)}."
    return pairs, None


class TpsMapperCanvas(FigureCanvas):
    """Unwarped (left) and Google Maps (right). 90° visual rotations; clicks mapped back to original coords."""
    PENDING_COLOR = "#000000"

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9, 4.5), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax_unwarped = self.fig.add_subplot(121)
        self.ax_map = self.fig.add_subplot(122)
        for ax in (self.ax_unwarped, self.ax_map):
            ax.set_axis_off()

        self.unwarped_image: Optional[Image.Image] = None
        self.map_image: Optional[Image.Image] = None
        self.unwarped_path = ""
        self.map_path = ""

        self.pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self._pending_unwarped: Optional[Tuple[float, float]] = None

        self.unwarped_angle_deg: float = 0.0
        self.map_angle_deg: float = 0.0

        
        self._cmap = cm.get_cmap("tab20")

        self.unwarped_flip_x = False  # mirror left↔right
        self.unwarped_flip_y = False  # mirror top↔bottom
        self.map_flip_x = False
        self.map_flip_y = False

        self._dragging_idx = None    # index of point being dragged
        self._dragging_side = None   # "unwarped" or "Google Maps"

        self._cid_click   = self.mpl_connect("button_press_event",   self._on_click)
        self._cid_motion  = self.mpl_connect("motion_notify_event",  self._on_motion)
        self._cid_release = self.mpl_connect("button_release_event", self._on_release)

    # Rotation (visual only)
    def rotate_unwarped(self, delta: int):
        self.unwarped_angle_deg = (self.unwarped_angle_deg + delta) % 360
        self.redraw()

    def rotate_map(self, delta: int):
        self.map_angle_deg = (self.map_angle_deg + delta) % 360
        self.redraw()

    def reset_unwarped(self): self.set_unwarped_angle(0)
    def reset_map(self): self.set_map_angle(0)
    def set_unwarped_angle(self, deg: float): self.unwarped_angle_deg = float(deg); self.redraw()
    def set_map_angle(self, deg: float): self.map_angle_deg = float(deg); self.redraw()

    def set_unwarped_flip_x(self, on: bool): self.unwarped_flip_x = bool(on); self.redraw()
    def set_unwarped_flip_y(self, on: bool): self.unwarped_flip_y = bool(on); self.redraw()
    def set_map_flip_x(self, on: bool): self.map_flip_x = bool(on); self.redraw()
    def set_map_flip_y(self, on: bool): self.map_flip_y = bool(on); self.redraw()

    def _hit_test(self, xdisp: float, ydisp: float, tol_px: float = 10.0):
        if xdisp is None or ydisp is None or not self.pairs:
            return None, None
        RT = self._unwarped_transform()
        MT = self._map_transform()
        unwarped_pts = np.array([r for (r, _) in self.pairs], dtype=float)
        map_pts = np.array([m for (_, m) in self.pairs], dtype=float)

        # Unwarped
        if unwarped_pts.size:
            unwarped_disp = RT.transform(unwarped_pts)  # (N,2)
            d = np.linalg.norm(unwarped_disp - np.array([xdisp, ydisp]), axis=1)
            ir = int(np.argmin(d))
            if d[ir] <= tol_px:
                return "unwarped", ir

        # Google Maps
        if map_pts.size:
            map_disp = MT.transform(map_pts)
            d = np.linalg.norm(map_disp - np.array([xdisp, ydisp]), axis=1)
            im = int(np.argmin(d))
            if d[im] <= tol_px:
                return "map", im

        return None, None



    # Transforms
    def _unwarped_transform(self) -> Affine2D:
        w, h = self._unwarped_size()
        cx, cy = w / 2.0, h / 2.0
        sx = -1.0 if self.unwarped_flip_x else 1.0
        sy = -1.0 if self.unwarped_flip_y else 1.0
        T = Affine2D()
        # center → apply scale (flip) → rotate → back
        T.translate(-cx, -cy).scale(sx, sy).rotate_deg(self.unwarped_angle_deg).translate(cx, cy)
        return T

    def _map_transform(self) -> Affine2D:
        w, h = self._map_size()
        cx, cy = w / 2.0, h / 2.0
        sx = -1.0 if self.map_flip_x else 1.0
        sy = -1.0 if self.map_flip_y else 1.0
        T = Affine2D()
        T.translate(-cx, -cy).scale(sx, sy).rotate_deg(self.map_angle_deg).translate(cx, cy)
        return T


    def _unwarped_size(self) -> Tuple[int, int]:
        return (self.unwarped_image.size if self.unwarped_image else (REQUIRED_W, REQUIRED_H))

    def _map_size(self) -> Tuple[int, int]:
        return (self.map_image.size if self.map_image else (REQUIRED_W, REQUIRED_H))

    def _set_axes_limits_to_fit(self, ax, transform: Affine2D, w: int, h: int):
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        tc = transform.transform(corners)
        xmin, ymin = tc.min(axis=0); xmax, ymax = tc.max(axis=0)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)

    # Load / set pairs
    def load_images(self, unwarped_path: str, map_path: str):
        self.unwarped_path = unwarped_path
        self.map_path = map_path
        self.unwarped_image = self._checked_open(unwarped_path)
        self.map_image = self._checked_open(map_path)
        self._pending_unwarped = None
        self.redraw()

    def _checked_open(self, path: str) -> Optional[Image.Image]:
        if not path or not os.path.isfile(path):
            return None
        with Image.open(path) as im:
            w, h = im.size
            if (w, h) != (REQUIRED_W, REQUIRED_H):
                raise ValueError(f"{os.path.basename(path)} must be {REQUIRED_W}x{REQUIRED_H}, got {w}x{h}.")
            return im.copy()

    def set_pairs(self, pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        self.pairs = pairs[:]
        self._pending_unwarped = None
        self.redraw()

    def get_pairs(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        return self.pairs[:]

    def clear_pairs(self):
        self.pairs.clear()
        self._pending_unwarped = None
        self.redraw()

    def undo_last(self):
        if self._pending_unwarped is not None:
            self._pending_unwarped = None
        elif self.pairs:
            self.pairs.pop()
        self.redraw()

    # Clicks → inverse-transform back to original coords before storing


    def _on_click(self, event):
        # Only respond inside one of our axes
        if event.inaxes not in (self.ax_unwarped, self.ax_map):
            return
        if event.xdata is None or event.ydata is None:
            return

        # If click is on/near an existing point → start dragging and DO NOT create/complete a pair
        side, idx = self._hit_test(event.xdata, event.ydata, tol_px=10.0)
        if side is not None:
            self._dragging_side = side
            self._dragging_idx  = idx
            # no redraw here; motion will redraw
            return

        # Otherwise, behave like your original click-to-pair logic
        x, y = float(event.xdata), float(event.ydata)
        if event.inaxes is self.ax_unwarped:
            rx, ry = self._unwarped_transform().inverted().transform((x, y))
            self._pending_unwarped = (rx, ry)
        else:
            if self._pending_unwarped is None:
                return
            mx, my = self._map_transform().inverted().transform((x, y))
            self.pairs.append((self._pending_unwarped, (mx, my)))
            self._pending_unwarped = None
        self.redraw()


    def _on_motion(self, event):
        if self._dragging_idx is None or self._dragging_side is None:
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        # Decide which side we’re dragging and use the correct inverse transform
        if self._dragging_side == "unwarped":
            invT = self._unwarped_transform().inverted()
            new_x, new_y = invT.transform((event.xdata, event.ydata))  # <-- unpack, no [0]
            unwarped, mp = self.pairs[self._dragging_idx]
            self.pairs[self._dragging_idx] = ((float(new_x), float(new_y)), mp)
        else:
            invT = self._map_transform().inverted()
            new_x, new_y = invT.transform((event.xdata, event.ydata))  # <-- unpack, no [0]
            unwarped, mp = self.pairs[self._dragging_idx]
            self.pairs[self._dragging_idx] = (unwarped, (float(new_x), float(new_y)))

        self.redraw()

    def _on_release(self, event):
        self._dragging_idx  = None
        self._dragging_side = None



    # Drawing
    def _pair_color(self, idx_zero_based: int):
        rgba = self._cmap(idx_zero_based % self._cmap.N)
        return rgba[:3]

    def redraw(self):
        self.ax_unwarped.clear(); self.ax_map.clear()
        for ax in (self.ax_unwarped, self.ax_map):
            ax.set_axis_off()

        # Unwarped
        if self.unwarped_image is not None:
            w, h = self._unwarped_size()
            T = self._unwarped_transform()
            self.ax_unwarped.imshow(self.unwarped_image, origin="upper", transform=T + self.ax_unwarped.transData)
            self._set_axes_limits_to_fit(self.ax_unwarped, T, w, h)
        else:
            self.ax_unwarped.text(0.5, 0.5, "No Unwarped image", ha="center", va="center", transform=self.ax_unwarped.transAxes)

        # Google Maps
        if self.map_image is not None:
            w, h = self._map_size()
            T = self._map_transform()
            self.ax_map.imshow(self.map_image, origin="upper", transform=T + self.ax_map.transData)
            self._set_axes_limits_to_fit(self.ax_map, T, w, h)
        else:
            self.ax_map.text(0.5, 0.5, "No Google Maps image", ha="center", va="center", transform=self.ax_map.transAxes)

        # Pairs
        RT = self._unwarped_transform(); MT = self._map_transform()
        for idx, (r, m) in enumerate(self.pairs):
            col = self._pair_color(idx)
            rx, ry = RT.transform(r); mx, my = MT.transform(m)
            self.ax_unwarped.plot(rx, ry, marker="o", ms=6, color=col)
            self.ax_map.plot(mx, my, marker="o", ms=6, color=col)
            self.ax_unwarped.text(rx + 4, ry + 4, str(idx + 1), color=col, fontsize=9)
            self.ax_map.text(mx + 4, my + 4, str(idx + 1), color=col, fontsize=9)

        if self._pending_unwarped is not None:
            rx, ry = RT.transform(self._pending_unwarped)
            self.ax_unwarped.plot(rx, ry, marker="x", ms=8, color=self.PENDING_COLOR)

        self.fig.tight_layout()
        self.draw_idle()



class PairerWidget(QtWidgets.QWidget):
    """UI: file open (1280x960 enforced), camera id, rotate ±90, save/load JSON, export .out, run mapping, color-coded table w/ trash."""

    pairsSaved = QtCore.Signal(str)

    def __init__(self, unwarped_guess: str = "", map_guess: str = ""):
        super().__init__()
        self.canvas = TpsMapperCanvas()

        # File & pair controls
        self.btn_open_unwarped = QtWidgets.QPushButton("Open Unwarped Image…")
        self.btn_open_map = QtWidgets.QPushButton("Open Google Maps Image…")
        self.btn_export_out = QtWidgets.QPushButton("Export .out")
        self.btn_save = QtWidgets.QPushButton("Save Pairs…")
        self.btn_load = QtWidgets.QPushButton("Load Pairs…")
        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_clear = QtWidgets.QPushButton("Clear")

        # Camera ID
        self.txt_cam = QtWidgets.QLineEdit()
        self.txt_cam.setPlaceholderText("Camera ID (e.g., 27)")
        self.lbl_status = QtWidgets.QLabel("Click Unwarped, then Google Maps to create a pair. Images must be 1280x960.")
        self.lbl_status.setStyleSheet("color: #555;")

        # Rotation buttons
        self.btn_unwarped_ccw = QtWidgets.QPushButton("Unwarped ⟲ 90°")
        self.btn_unwarped_cw  = QtWidgets.QPushButton("Unwarped ⟳ 90°")
        self.btn_unwarped_reset = QtWidgets.QPushButton("Reset Unwarped")
        self.btn_map_ccw = QtWidgets.QPushButton("Google Maps ⟲ 90°")
        self.btn_map_cw  = QtWidgets.QPushButton("Google Maps ⟳ 90°")
        self.btn_map_reset = QtWidgets.QPushButton("Reset Google Maps")

        self.chk_unwarped_flip_x = QtWidgets.QCheckBox("Flip Unwarped ↔")
        self.chk_unwarped_flip_y = QtWidgets.QCheckBox("Flip Unwarped ↕")
        self.chk_map_flip_x = QtWidgets.QCheckBox("Flip Google Maps ↔")
        self.chk_map_flip_y = QtWidgets.QCheckBox("Flip Google Maps ↕")


        # Table
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "Unwarped x", "Unwarped y", "Google Maps x", "Google Maps y", ""])
        for col in range(6):
            self.table.horizontalHeader().setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)


        # Layout
        files_row = QtWidgets.QHBoxLayout()
        files_row.addWidget(self.btn_open_unwarped)
        files_row.addWidget(self.btn_open_map)
        files_row.addStretch(1)
        files_row.addWidget(QtWidgets.QLabel("Camera ID:"))
        files_row.addWidget(self.txt_cam, 0)
        files_row.addStretch(1)
        files_row.addWidget(self.btn_undo)
        files_row.addWidget(self.btn_clear)

        rotate_row = QtWidgets.QHBoxLayout()
        rotate_row.addWidget(self.btn_unwarped_ccw)
        rotate_row.addWidget(self.btn_unwarped_cw)
        rotate_row.addWidget(self.btn_unwarped_reset)
        rotate_row.addSpacing(24)
        rotate_row.addWidget(self.btn_map_ccw)
        rotate_row.addWidget(self.btn_map_cw)
        rotate_row.addWidget(self.btn_map_reset)
        rotate_row.addStretch(1)

        rotate_row.addSpacing(24)
        rotate_row.addWidget(self.chk_unwarped_flip_x)
        rotate_row.addWidget(self.chk_unwarped_flip_y)
        rotate_row.addSpacing(12)
        rotate_row.addWidget(self.chk_map_flip_x)
        rotate_row.addWidget(self.chk_map_flip_y)


        io_row = QtWidgets.QHBoxLayout()
        io_row.addWidget(self.btn_load)
        io_row.addWidget(self.btn_save)
        io_row.addStretch(1)
        io_row.addWidget(self.btn_export_out)

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(files_row)
        v.addLayout(rotate_row)
        v.addWidget(self.canvas, stretch=1)
        v.addWidget(self.table, stretch=0)
        v.addLayout(io_row)
        v.addWidget(self.lbl_status)

        # Wire up
        self.btn_open_unwarped.clicked.connect(self._open_unwarped)
        self.btn_open_map.clicked.connect(self._open_map)
        self.btn_save.clicked.connect(self._save_pairs_json)
        self.btn_load.clicked.connect(self._load_pairs_any)
        self.btn_undo.clicked.connect(self.canvas.undo_last)
        self.btn_clear.clicked.connect(self._clear_pairs)
        self.btn_export_out.clicked.connect(self._export_out)

        self.btn_unwarped_ccw.clicked.connect(lambda: self.canvas.rotate_unwarped(-90))
        self.btn_unwarped_cw.clicked.connect(lambda: self.canvas.rotate_unwarped(+90))
        self.btn_unwarped_reset.clicked.connect(self.canvas.reset_unwarped)
        self.btn_map_ccw.clicked.connect(lambda: self.canvas.rotate_map(-90))
        self.btn_map_cw.clicked.connect(lambda: self.canvas.rotate_map(+90))
        self.btn_map_reset.clicked.connect(self.canvas.reset_map)

        self.chk_unwarped_flip_x.toggled.connect(self.canvas.set_unwarped_flip_x)
        self.chk_unwarped_flip_y.toggled.connect(self.canvas.set_unwarped_flip_y)
        self.chk_map_flip_x.toggled.connect(self.canvas.set_map_flip_x)
        self.chk_map_flip_y.toggled.connect(self.canvas.set_map_flip_y)


        self._sync_timer = QtCore.QTimer(self)
        self._sync_timer.setInterval(200)
        self._sync_timer.timeout.connect(self._sync_table)
        self._sync_timer.start()

        if unwarped_guess or map_guess:
            try:
                self.canvas.load_images(unwarped_guess, map_guess)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Open error", str(e))
            self._guess_cam_from_filenames()

    # public helper for Swiss Knife
    def suggest_images(self, unwarped_guess: str, map_guess: str):
        try:
            self.canvas.load_images(unwarped_guess, map_guess)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open error", str(e))
        self._guess_cam_from_filenames()

    # ---- UI handlers
    def _open_unwarped(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Unwarped image", ".", "Images (*.png *.jpg *.jpeg)")
        if fn:
            try:
                # enforce size
                with Image.open(fn) as im:
                    w, h = im.size
                if (w, h) != (REQUIRED_W, REQUIRED_H):
                    raise ValueError(f"Unwarped must be {REQUIRED_W}x{REQUIRED_H}, got {w}x{h}.")
                self.canvas.load_images(fn, self.canvas.map_path)
                self._guess_cam_from_filenames()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Unwarped error", str(e))

    def _open_map(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Google Maps image", ".", "Images (*.png *.jpg *.jpeg)")
        if fn:
            try:
                with Image.open(fn) as im:
                    w, h = im.size
                if (w, h) != (REQUIRED_W, REQUIRED_H):
                    raise ValueError(f"Google Maps must be {REQUIRED_W}x{REQUIRED_H}, got {w}x{h}.")
                self.canvas.load_images(self.canvas.unwarped_path, fn)
                self._guess_cam_from_filenames()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Google Maps error", str(e))

    def _guess_cam_from_filenames(self):
        # try to infer an integer from either filename
        for path in (self.canvas.map_path, self.canvas.unwarped_path):
            base = os.path.basename(path or "")
            for tok in base.replace("_", " ").replace("-", " ").split():
                if tok.isdigit():
                    self.txt_cam.setText(tok)
                    return

    def _save_pairs_json(self):
        if not self.canvas.unwarped_image or not self.canvas.map_image:
            QtWidgets.QMessageBox.warning(self, "No images", "Open Unwarped and Google Maps images first.")
            return
        pairs = self.canvas.get_pairs()
        data = {
            "unwarped_image": os.path.basename(self.canvas.unwarped_path),
            "map_image": os.path.basename(self.canvas.map_path),
            "unwarped_size": list(self.canvas.unwarped_image.size),
            "map_size": list(self.canvas.map_image.size),
            "pairs": [{"unwarped": [rx, ry], "map": [mx, my]} for (rx, ry), (mx, my) in pairs],
        }
        default = os.path.splitext(os.path.basename(self.canvas.unwarped_path or "unwarped"))[0] + "_tps_pairs.json"
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save pairs", default, "JSON (*.json)")
        if fn:
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.lbl_status.setText(f"Saved {len(pairs)} pairs to {fn}")

    def _load_pairs_json(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load pairs", ".", "JSON (*.json)")
        if not fn:
            return
        with open(fn, "r", encoding="utf-8") as f:
            data = json.load(f)
        # try to keep current images unless matching files exist locally
        pairs = []
        for item in data.get("pairs", []):
            r = item.get("unwarped", [None, None])
            m = item.get("map", [None, None])
            if None not in r and None not in m:
                pairs.append(((float(r[0]), float(r[1])), (float(m[0]), float(m[1]))))
        self.canvas.set_pairs(pairs)
        self.lbl_status.setText(f"Loaded {len(pairs)} pairs from {os.path.basename(fn)}")

    
    def _load_pairs_any(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load pairs", ".", "Pair files (*.json *.out *.txt);;JSON (*.json);;TPS OUT (*.out *.txt);;All files (*)"
        )
        if not fn:
            return

        ext = os.path.splitext(fn.lower())[1]
        try:
            if ext == ".json":
                with open(fn, "r", encoding="utf-8") as f:
                    data = json.load(f)

                pairs = []
                for item in data.get("pairs", []):
                    r = item.get("unwarped", [None, None])
                    m = item.get("map", [None, None])
                    if None not in r and None not in m:
                        pairs.append(((float(r[0]), float(r[1])), (float(m[0]), float(m[1]))))
                self.canvas.set_pairs(pairs)

                # Optional: restore rotations if present
                if "unwarped_angle_deg" in data:
                    self.canvas.set_unwarped_angle(data["unwarped_angle_deg"])
                if "map_angle_deg" in data:
                    self.canvas.set_map_angle(data["map_angle_deg"])

                self.lbl_status.setText(f"Loaded {len(pairs)} pairs from {os.path.basename(fn)}")

            else:
                # Treat as legacy .out (or .txt in .out format)
                pairs, warn = parse_legacy_out_pairs(fn)
                if not pairs:
                    QtWidgets.QMessageBox.warning(self, "No pairs found", "No valid rows parsed from this .out file.")
                    return
                self.canvas.set_pairs(pairs)
                if warn:
                    QtWidgets.QMessageBox.warning(self, "Count mismatch", warn)
                self.lbl_status.setText(f"Loaded {len(pairs)} pairs from legacy file {os.path.basename(fn)}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))


    def _export_out(self):
        # Validate
        if not self.canvas.unwarped_image or not self.canvas.map_image:
            QtWidgets.QMessageBox.warning(self, "No images", "Open Unwarped and Google Maps images first.")
            return
        if self.canvas.unwarped_image.size != (REQUIRED_W, REQUIRED_H) or self.canvas.map_image.size != (REQUIRED_W, REQUIRED_H):
            QtWidgets.QMessageBox.critical(self, "Size error", "Unwarped and Google Maps must be exactly 1280x960.")
            return
        cam_id = self.txt_cam.text().strip()
        if not cam_id.isdigit():
            QtWidgets.QMessageBox.warning(self, "Camera ID", "Enter a numeric Camera ID.")
            return
        pairs = self.canvas.get_pairs()
        if not pairs:
            QtWidgets.QMessageBox.warning(self, "No pairs", "Create at least one control point pair.")
            return

        default = f"{cam_id}_tps.out"
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export legacy .out", default, "TPS (*.out)")
        if not fn:
            return
        with open(fn, "w") as f:
            print(len(pairs), file=f)
            for (rx, ry), (mx, my) in pairs:
                f.write(f"{rx:.2f} {ry:.2f} {mx:.2f} {my:.2f}\n")

        self.pairsSaved.emit(fn)

        self.lbl_status.setText(f"Exported {len(pairs)} pairs to {fn}")

    def _clear_pairs(self):
        if QtWidgets.QMessageBox.question(self, "Clear pairs", "Remove all pairs?") == QtWidgets.QMessageBox.Yes:
            self.canvas.clear_pairs()

    # trash buttons & row colors
    def _delete_row(self, idx: int):
        pairs = self.canvas.get_pairs()
        if 0 <= idx < len(pairs):
            del pairs[idx]
            self.canvas.set_pairs(pairs)
            self.lbl_status.setText(f"Deleted pair #{idx+1}")

    def _sync_table(self):
        pairs = self.canvas.get_pairs()
        self.table.setRowCount(len(pairs))
        for i, (r, m) in enumerate(pairs):
            col = self.canvas._pair_color(i)
            qcolor = QtGui.QColor.fromRgbF(*col)

            idx_item = QtWidgets.QTableWidgetItem(str(i + 1))
            idx_item.setForeground(qcolor)
            self.table.setItem(i, 0, idx_item)

            rxi = QtWidgets.QTableWidgetItem(f"{r[0]:.1f}")
            ryi = QtWidgets.QTableWidgetItem(f"{r[1]:.1f}")
            mxi = QtWidgets.QTableWidgetItem(f"{m[0]:.1f}")
            myi = QtWidgets.QTableWidgetItem(f"{m[1]:.1f}")
            for it in (rxi, ryi, mxi, myi):
                it.setForeground(qcolor)
            self.table.setItem(i, 1, rxi)
            self.table.setItem(i, 2, ryi)
            self.table.setItem(i, 3, mxi)
            self.table.setItem(i, 4, myi)

            btn = QtWidgets.QToolButton()
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon)
            btn.setIcon(icon)
            btn.setToolTip("Delete this pair")
            btn.clicked.connect(lambda _, row=i: self._delete_row(row))
            self.table.setCellWidget(i, 5, btn)


def main():
    app = QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("TPS Mapper — Control Point Collector")
    widget = PairerWidget()
    win.setCentralWidget(widget)
    win.resize(1300, 780)
    win.show()
    app.exec()

if __name__ == "__main__":
    main()
