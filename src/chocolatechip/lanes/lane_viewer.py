"""
Intersection Lane Viewer (embeddable)
-------------------------------------
A widget that shows the chosen map image with overlays:
  • North/South/East/West reference lines
  • Dashed lane lines
  • Polygon boundary (default on)
Includes a legend. Auto-loads on intersection change.

Run standalone:
  python -m chocolatechip.lanes.lane_viewer
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from PIL import Image

from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from chocolatechip.MySQLConnector import MySQLConnector
from chocolatechip.intersections import (
    intersection_lookup,
    cam_lookup,
    IMAGE_BASE_URL,
    PICTURES,
    map_image_for,
)

COORD_PAIR_RE = re.compile(r"\(?\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)?")

def parse_pairs(s: str | None) -> List[Tuple[float, float]]:
    if not s:
        return []
    pairs: List[Tuple[float, float]] = []
    for mx in COORD_PAIR_RE.finditer(s):
        x = float(mx.group(1)); y = float(mx.group(2))
        pairs.append((x, y))
    return pairs

@dataclass
class IntersectionProps:
    intersection_id: int
    camera_id: int
    south_north: List[Tuple[float, float]]
    north_south: List[Tuple[float, float]]
    east_west: List[Tuple[float, float]]
    west_east: List[Tuple[float, float]]
    polygon: List[Tuple[float, float]]

class LaneViewerCanvas(FigureCanvas):
    LANE_COLOR = "#FFD400"   # yellow, dashed
    POLY_COLOR = "#9467bd"   # purple
    REF_COLORS = {
        "south_north": "#1f77b4",  # blue
        "north_south": "#ff7f0e",  # orange
        "east_west":   "#2ca02c",  # green
        "west_east":   "#d62728",  # red
    }

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self._bg_img = None

        self._lane_lines: List[Tuple[Tuple[float, float], Tuple[float, float], str]] = []
        self._ref_lines: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {
            "south_north": [], "north_south": [], "east_west": [], "west_east": []
        }
        self._polygon: List[Tuple[float, float]] = []

        self.show_lanes = True
        self.show_refs = True
        self.show_polygon = True

    def set_background_image(self, img_path: str):
        with Image.open(img_path) as im:
            self._bg_img = im.copy()
        self.redraw()

    def set_lanes(self, lanes_df: pd.DataFrame):
        lines = []
        for _, row in lanes_df.iterrows():
            p1 = (float(row["p1_x"]), float(row["p1_y"]))
            p2 = (float(row["p2_x"]), float(row["p2_y"]))
            label = str(row.get("name", ""))
            lines.append((p1, p2, label))
        self._lane_lines = lines
        self.redraw()

    def set_refs(self, props: IntersectionProps):
        def to_segments(pts: List[Tuple[float, float]]):
            segs = []
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    segs.append((pts[i], pts[i+1]))
            return segs
        self._ref_lines = {
            "south_north": to_segments(props.south_north),
            "north_south": to_segments(props.north_south),
            "east_west":   to_segments(props.east_west),
            "west_east":   to_segments(props.west_east),
        }
        self._polygon = props.polygon
        self.redraw()

    def clear(self):
        self._bg_img = None
        self._lane_lines = []
        self._ref_lines = {k: [] for k in self._ref_lines}
        self._polygon = []
        self.redraw()

    def _legend_handles(self) -> List[Line2D]:
        h: List[Line2D] = []
        if self.show_lanes:
            h.append(Line2D([0],[0], color=self.LANE_COLOR, lw=2, ls="--", label="Lane (dashed)"))
        if self.show_refs:
            h.extend([
                Line2D([0],[0], color=self.REF_COLORS["south_north"], lw=2, label="South→North ref"),
                Line2D([0],[0], color=self.REF_COLORS["north_south"], lw=2, label="North→South ref"),
                Line2D([0],[0], color=self.REF_COLORS["east_west"],   lw=2, label="East→West ref"),
                Line2D([0],[0], color=self.REF_COLORS["west_east"],   lw=2, label="West→East ref"),
            ])
        if self.show_polygon:
            h.append(Line2D([0],[0], color=self.POLY_COLOR, lw=2, label="Intersection polygon"))
        return h

    def redraw(self):
        self.ax.clear()
        self.ax.set_axis_off()

        if self._bg_img is not None:
            self.ax.imshow(self._bg_img)
            w, h = self._bg_img.size
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0)

        if self.show_polygon and self._polygon:
            pts = self._polygon
            if pts and pts[0] != pts[-1]:
                pts = pts + [pts[0]]
            if pts:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                self.ax.plot(xs, ys, linestyle="-", linewidth=1.5, color=self.POLY_COLOR)

        if self.show_refs:
            for key, segs in self._ref_lines.items():
                col = self.REF_COLORS.get(key, "#000000")
                for (x1,y1),(x2,y2) in segs:
                    self.ax.plot([x1,x2],[y1,y2], linestyle="-", linewidth=1.5, color=col)

        if self.show_lanes:
            for (p1, p2, _label) in self._lane_lines:
                (x1,y1),(x2,y2) = p1,p2
                self.ax.plot([x1,x2],[y1,y2], linestyle="--", linewidth=1.5, color=self.LANE_COLOR)

        h = self._legend_handles()
        if h:
            leg = self.ax.legend(h, [hh.get_label() for hh in h],
                                 loc="lower left", framealpha=0.85,
                                 facecolor="white", edgecolor="black", fontsize=9)
            for txt in leg.get_texts():
                txt.set_color("black")

        self.fig.tight_layout()
        self.draw_idle()


class LaneViewerWidget(QtWidgets.QWidget):
    """
    Embeddable viewer widget. Emits intersectionChanged(iid, cam_id, map_file)
    when the selection changes (for Swiss Knife to inform TPS tab).
    """
    intersectionChanged = QtCore.Signal(int, int, str)

    def __init__(self):
        super().__init__()
        self.db = MySQLConnector()

        # Controls
        self.cmb_intersection = QtWidgets.QComboBox()
        self.chk_lanes = QtWidgets.QCheckBox("Show Lanes (dashed)")
        self.chk_refs = QtWidgets.QCheckBox("Show NS/EW refs")
        self.chk_polygon = QtWidgets.QCheckBox("Show polygon")
        self.chk_lanes.setChecked(True)
        self.chk_refs.setChecked(True)
        self.chk_polygon.setChecked(True)

        # Canvas
        self.canvas = LaneViewerCanvas()

        # Layout
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Intersection:"))
        controls.addWidget(self.cmb_intersection, stretch=1)
        controls.addStretch(1)
        controls.addWidget(self.chk_lanes)
        controls.addWidget(self.chk_refs)
        controls.addWidget(self.chk_polygon)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addLayout(controls)
        vbox.addWidget(self.canvas, stretch=1)

        # Populate
        self._id_by_name: Dict[str, int] = {}
        for iid, name in intersection_lookup.items():
            self._id_by_name[name] = iid
            self.cmb_intersection.addItem(name)

        # Prefetch images
        self._ensure_images_downloaded()

        # Signals
        self.cmb_intersection.currentIndexChanged.connect(lambda _: self._load_current())
        self.chk_lanes.toggled.connect(self._toggle)
        self.chk_refs.toggled.connect(self._toggle)
        self.chk_polygon.toggled.connect(self._toggle)

        # Initial load
        if self.cmb_intersection.count():
            self._load_current()

    # Public getter for current ids
    def current_ids(self) -> Tuple[Optional[int], Optional[int]]:
        name = self.cmb_intersection.currentText()
        if not name:
            return None, None
        iid = self._id_by_name[name]
        cam_id = cam_lookup.get(iid)
        return iid, cam_id

    # ---- UI callbacks
    def _toggle(self):
        self.canvas.show_lanes = self.chk_lanes.isChecked()
        self.canvas.show_refs = self.chk_refs.isChecked()
        self.canvas.show_polygon = self.chk_polygon.isChecked()
        self.canvas.redraw()

    def _load_current(self):
        name = self.cmb_intersection.currentText()
        if not name:
            return
        iid = self._id_by_name[name]
        cam_id = cam_lookup.get(iid)

        try:
            img_file = map_image_for(iid, camera_id=cam_id)
        except KeyError as e:
            QtWidgets.QMessageBox.warning(self, "Image mapping missing", str(e))
            return

        if not os.path.isfile(img_file):
            self._download_image(img_file)

        if os.path.isfile(img_file):
            self.canvas.set_background_image(img_file)
        else:
            QtWidgets.QMessageBox.warning(self, "Image missing", f"Could not load {img_file}")

        lanes_df = self._fetch_lanes(iid, cam_id)
        props = self._fetch_intersection_props(iid, cam_id)
        self.canvas.set_lanes(lanes_df)
        self.canvas.set_refs(props or IntersectionProps(iid, cam_id or -1, [], [], [], [], []))

        # Notify listeners (Swiss Knife)
        self.intersectionChanged.emit(int(iid), int(cam_id) if cam_id is not None else -1, img_file)

    # ---- Data access
    def _fetch_lanes(self, intersection_id: int, camera_id: int) -> pd.DataFrame:
        sql = (
            "SELECT name, p1_x, p1_y, p2_x, p2_y, width, mapdirection, trafficdirection "
            "FROM Lanes WHERE intersection_id = %s AND camera_id = %s"
        )
        with self.db._connect() as conn:
            df = pd.read_sql(sql, con=conn, params=[intersection_id, camera_id])
        return df

    def _fetch_intersection_props(self, intersection_id: int, camera_id: int) -> Optional[IntersectionProps]:
        sql = (
            "SELECT intersection_id, camera_id, south_north, north_south, east_west, west_east, polygon "
            "FROM IntersectionProperties WHERE intersection_id = %s AND camera_id = %s LIMIT 1"
        )
        with self.db._connect() as conn:
            df = pd.read_sql(sql, con=conn, params=[intersection_id, camera_id])
        if df.empty:
            return None
        r = df.iloc[0]
        return IntersectionProps(
            intersection_id=int(r["intersection_id"]),
            camera_id=int(r["camera_id"]),
            south_north=parse_pairs(str(r.get("south_north", ""))),
            north_south=parse_pairs(str(r.get("north_south", ""))),
            east_west=parse_pairs(str(r.get("east_west", ""))),
            west_east=parse_pairs(str(r.get("west_east", ""))),
            polygon=parse_pairs(str(r.get("polygon", ""))),
        )

    # ---- Image helpers
    def _ensure_images_downloaded(self):
        for picture in PICTURES:
            if not os.path.isfile(picture):
                self._download_image(picture)

    def _download_image(self, picture: str) -> bool:
        try:
            url = f"{IMAGE_BASE_URL}{picture}"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            with open(picture, "wb") as f:
                f.write(resp.content)
            return True
        except Exception:
            return False


def main():
    app = QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Intersection Lane Viewer")
    widget = LaneViewerWidget()
    win.setCentralWidget(widget)
    win.resize(1200, 850)
    win.show()
    app.exec()

if __name__ == "__main__":
    main()
