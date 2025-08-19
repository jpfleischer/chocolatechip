# chocolatechip/lanes/unwrapper.py
from __future__ import annotations

from pathlib import Path
import math
import cv2
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets


# --------- math helpers (self-contained) ----------
def _RzRyRx_intrinsic(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yz, yp, xr = map(math.radians, [yaw_deg, pitch_deg, roll_deg])
    cz, sz = math.cos(yz), math.sin(yz)
    cp, sp = math.cos(yp), math.sin(yp)
    cr, sr = math.cos(xr), math.sin(xr)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def build_unwarp_maps(
    H: int, W: int,
    f: float, cx: float, cy: float,
    yaw: float, pitch: float, roll: float,
    z_const: float,
    model: str = "equisolid",
    rmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create OpenCV remap maps (mapx, mapy) for fisheye → ground view."""
    yy, xx = np.indices((H, W), dtype=np.float32)
    grnd = np.stack([yy - cy, xx - cx, np.full((H, W), z_const, np.float32)], axis=-1)

    R = _RzRyRx_intrinsic(yaw, pitch, roll)
    U = grnd @ R.T
    U /= (np.linalg.norm(U, axis=-1, keepdims=True) + 1e-12)

    Ux, Uy, Uz = U[..., 0], U[..., 1], U[..., 2]

    # theta = arccos(-Uz) (optical axis along -Z)
    cos_theta = np.clip(-Uz, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if model == "equisolid":
        # r = 2 f sin(theta/2) == f * sqrt(2*(1 + Uz))
        r = f * np.sqrt(np.clip(2.0 * (1.0 + Uz), 0.0, 4.0))
    elif model == "equidistant":
        # r = f * theta
        r = f * theta
    else:
        raise ValueError("model must be 'equisolid' or 'equidistant'")

    den = np.hypot(Ux, Uy) + 1e-12
    mapx = cx + r * (Ux / den)
    mapy = cy + r * (Uy / den)

    if rmax is None:
        rmax = min(cx, cy, W - cx, H - cy)
    mask = r <= float(rmax)
    mapx = np.where(mask, mapx, -1).astype(np.float32)
    mapy = np.where(mask, mapy, -1).astype(np.float32)
    return mapx, mapy


# --------- small utils ----------
def _cv_to_qpixmap(img_bgr: np.ndarray) -> QtGui.QPixmap:
    if img_bgr is None:
        return QtGui.QPixmap()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    qimg = QtGui.QImage(img_rgb.data, w, h, img_rgb.strides[0], QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


class UnwrapperWidget(QtWidgets.QWidget):
    """GUI panel to unwrap a fisheye image → ground-view + save maps."""

    paramsChanged = QtCore.Signal(dict)   # emits dict of current params
    imageLoaded   = QtCore.Signal(str)    # emits path of currently loaded image
    unwrappedReady= QtCore.Signal(str, str)  # emits (png_path, npz_path) when saved


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("UnwrapperWidget")

        self._src_path: Path | None = None
        self._src_img: np.ndarray | None = None
        self._preview_img: np.ndarray | None = None
        self._last_maps: tuple[np.ndarray,np.ndarray] | None = None

        # --- Left controls ---
        self.btn_load  = QtWidgets.QPushButton("Load Fisheye…")
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("No file chosen")
        self.path_edit.setReadOnly(True)

        # Projection model
        self.model = QtWidgets.QComboBox()
        self.model.addItems(["equisolid", "equidistant"])

        # Numeric params
        def dspin(minv,maxv,step,defv,dec=3):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(minv,maxv); s.setSingleStep(step); s.setValue(defv); s.setDecimals(dec)
            return s

        self.focal = dspin(1, 5000, 1, 355.0, 3)
        self.cx    = dspin(0, 10000, 0.1, 0.0, 2)
        self.cy    = dspin(0, 10000, 0.1, 0.0, 2)
        self.yaw   = dspin(-360, 360, 1, 125.0, 3)
        self.pitch = dspin(-360, 360, 1, 0.0,   3)
        self.roll  = dspin(-360, 360, 1, 0.0,   3)
        self.z     = dspin(-10000, 10000, 1, -80.0, 3)
        self.rmax  = dspin(0, 10000, 1, 0.0, 2); self.rmax.setSpecialValueText("auto (0)")

        self.btn_center  = QtWidgets.QPushButton("Center cx, cy")

        # Preview toggle
        self.btn_preview = QtWidgets.QPushButton("Preview")
        self.btn_preview.setCheckable(True)
        self.btn_preview.setChecked(False)
        self.btn_preview.toggled.connect(self._on_preview_toggled)

        # Save button (CREATE before connecting)
        self.btn_save = QtWidgets.QPushButton("Save PNG + Maps")

        # --- Form layout (now safe to reference spinboxes) ---
        grid = QtWidgets.QFormLayout()
        grid.addRow(self.btn_load, self.path_edit)
        grid.addRow("Model:", self.model)
        grid.addRow("Focal (px):", self.focal)
        grid.addRow("cx (px):", self.cx)
        grid.addRow("cy (px):", self.cy)
        grid.addRow("Yaw (°):", self.yaw)
        grid.addRow("Pitch (°):", self.pitch)
        grid.addRow("Roll (°):", self.roll)
        grid.addRow("Z depth (px):", self.z)
        grid.addRow("rmax (px):", self.rmax)

        # Help text under rmax
        self.help_label = QtWidgets.QLabel(
            "It is best to leave the defaults. Just load a fisheye image and click "
            "<b>Preview</b> to see how it would look, then save it if it is good."
        )
        self.help_label.setWordWrap(True)
        self.help_label.setStyleSheet("color: gray; font-size: 10pt;")
        grid.addRow("", self.help_label)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.btn_center)
        btns.addWidget(self.btn_preview)
        btns.addWidget(self.btn_save)

        left = QtWidgets.QVBoxLayout()
        left.addLayout(grid)
        left.addStretch(1)
        left.addLayout(btns)

        left_box = QtWidgets.QGroupBox("Parameters")
        left_box.setLayout(left)

        # --- Right preview ---
        self.preview = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setBackgroundRole(QtGui.QPalette.ColorRole.Base)
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                QtWidgets.QSizePolicy.Policy.Expanding)
        self.preview.setStyleSheet("QLabel { background: #111; border: 1px solid #333; }")

        # --- Layout ---
        main = QtWidgets.QHBoxLayout(self)
        main.addWidget(left_box, 0)
        main.addWidget(self.preview, 1)

        # --- Signals ---
        self.btn_load.clicked.connect(self._choose_image)
        self.btn_center.clicked.connect(self._auto_center)
        self.btn_save.clicked.connect(self._do_save)
        self.paramsChanged.connect(self._on_params_changed)
        for w in [self.focal,self.cx,self.cy,self.yaw,self.pitch,self.roll,self.z,self.rmax]:
            w.valueChanged.connect(self._emit_params)
        self.model.currentTextChanged.connect(self._emit_params)

        self.setAcceptDrops(True)


    # ---- Drag & drop ----
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent) -> None:
        urls = e.mimeData().urls()
        if urls:
            self._load_path(Path(urls[0].toLocalFile()))

    # ---- UI helpers ----
    def _choose_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose fisheye image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self._load_path(Path(path))

    def _load_path(self, p: Path):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            QtWidgets.QMessageBox.critical(self, "Load failed", f"Could not read: {p}")
            return
        self._src_path = p
        self._src_img  = img
        H, W = img.shape[:2]
        self.path_edit.setText(str(p))
        # If cx/cy are 0 (initial), center them to image
        if self.cx.value() == 0.0 and self.cy.value() == 0.0:
            self.cx.setValue(W * 0.5)
            self.cy.setValue(H * 0.5)
        self._show_preview(img)
        self.imageLoaded.emit(str(p))

        if self.btn_preview.isChecked():
            img2 = self._compute_unwarp()
            if img2 is not None:
                self._show_preview(img2)


    def _auto_center(self):
        if self._src_img is None:
            return
        H, W = self._src_img.shape[:2]
        self.cx.setValue(W * 0.5)
        self.cy.setValue(H * 0.5)

    def _collect_params(self) -> dict:
        return dict(
            model=self.model.currentText(),
            focal=float(self.focal.value()),
            cx=float(self.cx.value()),
            cy=float(self.cy.value()),
            yaw=float(self.yaw.value()),
            pitch=float(self.pitch.value()),
            roll=float(self.roll.value()),
            z=float(self.z.value()),
            rmax=float(self.rmax.value()),
        )
    
    def _compute_unwarp(self):
        """Compute maps + unwrapped image using current params. Updates caches and returns image."""
        if self._src_img is None:
            return None
        params = self._collect_params()
        src = self._src_img
        H, W = src.shape[:2]
        rmax = None if params["rmax"] <= 0.0 else params["rmax"]

        mapx, mapy = build_unwarp_maps(
            H=H, W=W,
            f=params["focal"], cx=params["cx"], cy=params["cy"],
            yaw=params["yaw"], pitch=params["pitch"], roll=params["roll"],
            z_const=params["z"], model=params["model"], rmax=rmax
        )
        self._last_maps = (mapx, mapy)

        unwarped = cv2.remap(
            src, mapx, mapy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        self._preview_img = unwarped
        return unwarped


    def _on_preview_toggled(self, checked: bool):
        if self._src_img is None:
            QtWidgets.QMessageBox.information(self, "No image", "Load a fisheye image first.")
            # revert toggle if no image loaded
            self.btn_preview.blockSignals(True)
            self.btn_preview.setChecked(False)
            self.btn_preview.blockSignals(False)
            self.btn_preview.setText("Preview")
            return

        if checked:
            self.btn_preview.setText("Show Original")
            img = self._compute_unwarp()
            if img is not None:
                self._show_preview(img)
        else:
            self.btn_preview.setText("Preview")
            self._show_preview(self._src_img)


    def _on_params_changed(self, *_):
        """Recompute live if preview is toggled on."""
        if getattr(self, "btn_preview", None) and self.btn_preview.isChecked() and self._src_img is not None:
            img = self._compute_unwarp()
            if img is not None:
                self._show_preview(img)


    def _emit_params(self, *args):
        self.paramsChanged.emit(self._collect_params())



    def _show_preview(self, img_bgr: np.ndarray):
        pm = _cv_to_qpixmap(img_bgr)
        # Fit-to-label
        scaled = pm.scaled(self.preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                           QtCore.Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(scaled)

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        if self._preview_img is not None:
            self._show_preview(self._preview_img)
        elif self._src_img is not None:
            self._show_preview(self._src_img)

    def _do_save(self):
        if self._src_img is None:
            QtWidgets.QMessageBox.information(self, "No image", "Load a fisheye image first.")
            return
        if self._preview_img is None or self._last_maps is None:
            if self._compute_unwarp() is None:
                return

        stem = (self._src_path.stem if self._src_path else "frame")
        out_png, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save unwarped PNG", f"{stem}_unwarped.png", "PNG Image (*.png)"
        )
        if not out_png:
            return
        out_npz, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save unwarp maps (.npz)", f"{stem}_unwarp_maps.npz", "NumPy Zip (*.npz)"
        )
        if not out_npz:
            return

        ok = cv2.imwrite(out_png, self._preview_img)
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Save failed", f"Could not write {out_png}")
            return

        mapx, mapy = self._last_maps
        params = self._collect_params()
        np.savez_compressed(
            out_npz,
            mapx=mapx, mapy=mapy,
            H=self._src_img.shape[0], W=self._src_img.shape[1],
            cx=params["cx"], cy=params["cy"], f=params["focal"],
            model=params["model"],
            yaw=params["yaw"], pitch=params["pitch"], roll=params["roll"], z=params["z"]
        )

        self.unwrappedReady.emit(out_png, out_npz)
        QtWidgets.QMessageBox.information(self, "Saved", f"PNG → {out_png}\nMaps → {out_npz}")

    # ---- Optional API for external seeding ----
    def load_image(self, path: str | Path):
        self._load_path(Path(path))

    def set_params(self, **kwargs):
        if "model" in kwargs:
            idx = self.model.findText(str(kwargs["model"]))
            if idx >= 0:
                self.model.setCurrentIndex(idx)
        for k, widget in dict(focal=self.focal, cx=self.cx, cy=self.cy,
                              yaw=self.yaw, pitch=self.pitch, roll=self.roll,
                              z=self.z, rmax=self.rmax).items():
            if k in kwargs and kwargs[k] is not None:
                widget.setValue(float(kwargs[k]))
