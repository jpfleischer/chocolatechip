# chocolatechip/lanes/intersection_configurator.py
from __future__ import annotations

from PySide6 import QtWidgets
from chocolatechip.lanes.lane_viewer import LaneViewerWidget
from chocolatechip.lanes.pairer import PairerWidget
from chocolatechip.lanes.unwrapper import UnwrapperWidget
from chocolatechip.lanes.mapper import MapperWidget
from chocolatechip.intersections import cam_lookup, map_image_for

import glob


class SwissKnife(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChocolateChip — Swiss Army Knife")
        self.resize(1500, 950)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # 0: Viewer
        self.viewer = LaneViewerWidget()
        self.tabs.addTab(self.viewer, "Viewer")

        # 1: Unwrapper
        self.unwrapper = UnwrapperWidget()
        self.tabs.insertTab(1, self.unwrapper, "Unwrapper")

        # 2: Point Pairer (point pairing)
        self.tps = PairerWidget()
        self.tabs.addTab(self.tps, "Point Pairer")

        # 3: Mapper (export .map)
        self.mapper = MapperWidget()
        self.tabs.addTab(self.mapper, "Mapper")

        # Wiring
        self.viewer.intersectionChanged.connect(self._suggest_to_tps)
        self.tabs.currentChanged.connect(self._maybe_seed_tps)
        self.tabs.currentChanged.connect(self._maybe_seed_mapper)

        # Optional: if TPS widget emits a signal when a .out is saved, forward it here.
        # We'll try to connect dynamically; if not present, ignore.
        try:
            self.tps.pairsSaved.connect(self._on_pairs_saved)  # signature: (out_path: str)
        except Exception:
            pass

        # Optional: when Unwrapper saves, suggest rectified to Mapper
        try:
            self.unwrapper.unwrappedReady.connect(self._on_unwrapped_saved)  # (png_path, npz_path)
        except Exception:
            pass

    def _suggest_to_tps(self, iid: int, cam_id: int, map_file: str):
        unwarped_guess = ""
        if cam_id and cam_id > 0:
            # look for any file like "27*unwarped.png"
            pattern = f"{int(cam_id)}*unwarped.png"
            matches = glob.glob(pattern)
            if matches:
                # take the first match (or you could sort and pick best)
                unwarped_guess = matches[0]
        self.tps.suggest_images(unwarped_guess, map_file)

    def _maybe_seed_tps(self, idx: int):
        if self.tabs.tabText(idx) != "Point Pairer":
            return
        iid, cam_id = self.viewer.current_ids()
        if iid is None or cam_id is None:
            return
        try:
            map_file = map_image_for(iid, cam_id)
        except Exception:
            map_file = f"{int(cam_id)}*Map.png" if cam_id else ""

        unwarped_guess = ""
        if cam_id:
            pattern = f"{int(cam_id)}*unwarped.png"
            matches = glob.glob(pattern)
            if matches:
                unwarped_guess = matches[0]

        self.tps.suggest_images(unwarped_guess, map_file)

    def _maybe_seed_mapper(self, idx: int):
        if self.tabs.tabText(idx) != "Mapper":
            return

        iid, cam_id = self.viewer.current_ids()
        if cam_id is None:
            return

        cam = int(cam_id)

        # Rectified (unwarped) guess via glob: e.g., "24*unwarped.png"
        rect_matches = sorted(glob.glob(f"{cam}*unwarped.png"))
        rectified = rect_matches[0] if rect_matches else ""

        # TPS OUT guess via glob: e.g., "24*tps.out"
        tps_matches = sorted(glob.glob(f"{cam}*tps.out"))
        tps_out = tps_matches[0] if tps_matches else ""

        # Map image: prefer official lookup; if that fails, glob "24*Map.png"
        try:
            map_img = map_image_for(iid, cam_id)
        except Exception:
            map_matches = sorted(glob.glob(f"{cam}*Map.png"))
            map_img = map_matches[0] if map_matches else ""

        self.mapper.suggest_files(rectified, map_img, tps_out, out_dir="output_align")


    def _on_unwrapped_saved(self, png_path: str, _npz_path: str):
        """When Unwrapper saves, prefill Mapper's rectified path."""
        self.mapper.suggest_files(rectified=png_path, map_img=None, tps_out=None, out_dir=None)

    def _on_pairs_saved(self, out_path: str):
        """When TPS saves a .out file, prefill Mapper's tps_out."""
        self.mapper.suggest_files(rectified=None, map_img=None, tps_out=out_path, out_dir=None)


def main():
    app = QtWidgets.QApplication([])
    win = SwissKnife()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
