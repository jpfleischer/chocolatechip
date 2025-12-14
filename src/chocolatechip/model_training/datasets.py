#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, json, shutil, hashlib, tempfile, tarfile, zipfile, subprocess
from dataclasses import asdict
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .profiles import DatasetSpec

MARKER_NAME = ".download_complete.json"

def _eprint(*a, **k): print(*a, file=sys.stderr, **k)

def _marker_path(root: Path) -> Path:
    return root / MARKER_NAME

def already_prepared(root: Path) -> bool:
    return root.is_dir() and _marker_path(root).is_file()

def _sha256(path: Path, bufsize: int = 2**20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()

def _flat_top(spec) -> str | None:
    fd = getattr(spec, "flat_dir", None)
    if not fd:
        return None
    return fd.strip("/").split("/")[0]

# ---- NEW: expected dirs = sets ∪ neg_subdirs --------------------------------
def _expected_dirs(spec) -> set[str]:
    """Dirs we must preserve & consider part of the dataset root."""
    return set(spec.sets) | set(getattr(spec, "neg_subdirs", []) or [])

def _looks_ready(root: Path, spec) -> bool:
    try:
        # if a flat cache is specified, that's enough
        if getattr(spec, "flat_dir", None):
            if (root / spec.flat_dir).is_dir():
                return True
        expected = set(spec.sets) | set(getattr(spec, "neg_subdirs", []) or [])
        return all((root / d).is_dir() for d in expected)
    except Exception:
        return False


def _find_dir_with_sets(root: Path, spec) -> Path | None:
    expected = set(spec.sets) | set(getattr(spec, "neg_subdirs", []) or [])
    flat_rel = getattr(spec, "flat_dir", None)
    candidates = [root] + [p for p in root.rglob("*")
                           if p.is_dir() and p.relative_to(root).parts
                           and len(p.relative_to(root).parts) <= 3]
    for c in candidates:
        ok_sets = all((c / d).is_dir() for d in expected) if expected else False
        ok_flat = (flat_rel is not None) and (c / flat_rel).is_dir()
        if ok_sets or ok_flat:                 # <--- NEW
            return c
    return None


def _promote_sets_to_root(root: Path, src: Path, spec) -> None:
    if src == root:
        return
    expected = set(spec.sets) | set(getattr(spec, "neg_subdirs", []) or [])
    flat_top = _flat_top(spec)                 # <--- NEW
    if flat_top:
        expected.add(flat_top)                 # <--- include darkmark_image_cache

    wanted_files = {spec.names, f"{spec.prefix}.data", f"{spec.prefix}.cfg", "license.txt"}
    wanted = expected | wanted_files

    for item in src.iterdir():
        name = item.name
        if name in wanted:
            shutil.move(str(item), root / name)

    if not _looks_ready(root, spec):
        # still missing (e.g., flat_dir nested deeper); move everything
        for item in list(src.iterdir()):
            shutil.move(str(item), root / item.name)

    shutil.rmtree(src, ignore_errors=True)

# -----------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _eprint(f"[download] {url} -> {dest}")
    req = Request(url, headers={"User-Agent": "chocolatechip-datasets/1.0"})
    with urlopen(req) as r, open(dest, "wb") as f:
        total = r.length if hasattr(r, "length") else None
        copied = 0
        while True:
            buf = r.read(1024 * 1024)
            if not buf: break
            f.write(buf); copied += len(buf)
            if total:
                pct = copied * 100.0 / total
                _eprint(f"\r  {copied/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)", end="")
    _eprint()

def _extract(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    _eprint(f"[extract] {archive.name} -> {dest_dir}")
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(dest_dir)
        return
    # tarballs (and friends)
    try:
        with tarfile.open(archive, "r:*") as t:
            t.extractall(dest_dir)
        return
    except tarfile.ReadError:
        pass
    # fallback: plain file copy
    shutil.copy2(archive, dest_dir / archive.name)

def _mark_complete(root: Path, meta: dict):
    meta = dict(meta)
    meta["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (_marker_path(root)).write_text(json.dumps(meta, indent=2), encoding="utf-8")

class DirLock:
    def __init__(self, base: Path):
        self.path = base / ".lock_dataset_manager"
    def __enter__(self):
        # ensure parent exists (e.g., /workspace/LegoGears_v2)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.mkdir(exist_ok=False)
        return self
    def __exit__(self, exc_type, exc, tb):
        try: self.path.rmdir()
        except Exception: pass

def ensure_download_once(spec: DatasetSpec, *, force: bool = False) -> Path:
    """
    One-time download/extract into spec.root (absolute or relative to CWD).
    Skips if marker exists, unless force=True.
    Returns the dataset root Path.
    """
    root = Path(spec.root).resolve()

        # -------- Local-only mode: fail fast, no writes, no marker --------
    if getattr(spec, "require_existing", False):
        if not root.exists():
            raise FileNotFoundError(f"[local-only] dataset root not found: {root}")
        # Ok if either sets/flat_dir exist OR a .data file already exists
        def _has_expected():
            try:
                if getattr(spec, "flat_dir", None) and (root / spec.flat_dir).is_dir():
                    return True
                expected = set(spec.sets) | set(getattr(spec, "neg_subdirs", []) or [])
                if expected and all((root / d).is_dir() for d in expected):
                    return True
            except Exception:
                pass
            return False
        has_data = any(p.is_file() for p in root.glob("*.data"))
        if not (_has_expected() or has_data):
            raise RuntimeError(
                f"[local-only] dataset not ready at {root}. "
                f"Expected sets/flat_dir present OR an existing .data file."
            )
        _eprint(f"[ok] local-only dataset present at {root}")
        return root

    # make both parent and root to support lock creation
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    if not force and already_prepared(root) and _looks_ready(root, spec):
        _eprint(f"[skip] dataset ready at {root}")
        return root

    # marker exists but expected dirs missing → repair
    if already_prepared(root) and not _looks_ready(root, spec):
        _eprint(f"[warn] marker exists but expected dirs missing; repairing {root}")
        force = True

    with DirLock(root):
        if force and root.exists():
            _eprint(f"[force] removing {root}")
            shutil.rmtree(root, ignore_errors=True)

        url = spec.url
        meta = {"profile_dataset": asdict(spec)}
        if not url:
            _eprint(f"[info] no URL provided; assuming files already present at {root}")
            root.mkdir(parents=True, exist_ok=True)
            _mark_complete(root, meta)
            return root

        tmpdir = Path(tempfile.mkdtemp(prefix="dsdl_"))
        try:
            fname = url.split("?")[0].rsplit("/", 1)[-1] or "dataset.download"
            archive = tmpdir / fname
            try:
                _download(url, archive)
            except (HTTPError, URLError) as ex:
                raise RuntimeError(f"download failed: {ex}")

            if spec.sha256:
                got = _sha256(archive)
                if got.lower() != spec.sha256.lower():
                    raise RuntimeError(f"checksum mismatch: expected {spec.sha256}, got {got}")
                meta["sha256"] = got

            _extract(archive, root)

            # flatten trivial single-dir case
            children = [c for c in root.iterdir() if c.name != MARKER_NAME]
            if len(children) == 1 and children[0].is_dir():
                inner = children[0]
                for item in inner.iterdir():
                    shutil.move(str(item), root / item.name)
                shutil.rmtree(inner, ignore_errors=True)

            # if expected dirs still aren’t at root/, find and promote them
            if not _looks_ready(root, spec):
                cand = _find_dir_with_sets(root, spec)
                if cand is not None:
                    _eprint(f"[normalize] promoting expected dirs from {cand} -> {root}")
                    _promote_sets_to_root(root, cand, spec)

            # sanity check; if still not ready, show what we extracted
            if not _looks_ready(root, spec):
                _eprint("[error] expected dirs missing after extract/normalize")
                try:
                    tree = "\n".join(str(p) for p in root.rglob("*") if p.is_dir())[:2000]
                    _eprint(tree)
                except Exception:
                    pass
                raise RuntimeError(f"Extracted dataset does not contain {_expected_dirs(spec)} at {root}")

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        _mark_complete(root, meta)
        _eprint(f"[ok] dataset ready at {root}")
        return root

def ensure_splits(spec: DatasetSpec) -> None:
    root = Path(spec.root).resolve()
    train = root / f"{spec.prefix}_train.txt"
    valid = root / f"{spec.prefix}_valid.txt"
    data  = root / f"{spec.prefix}.data"
    if train.is_file() and valid.is_file() and data.is_file():
        _eprint("[split] existing split detected; skipping")
        return

    # Use the correct field from DatasetSpec
    seed_val = getattr(spec, "split_seed", 9001)

    if getattr(spec, "flat_dir", None):
        cmd = [
            sys.executable, "-m", "chocolatechip.model_training.dataset_setup",
            "--root", str(root),
            "--flat-dir", spec.flat_dir,
            "--classes", str(spec.classes),
            "--names", spec.names,
            "--prefix", spec.prefix,
            "--val-frac", "0.20",
            "--seed", str(seed_val),
            "--exts", *[e.lower() for e in spec.exts],
        ]
    else:
        all_sets = list(set(spec.sets) | set(getattr(spec, "neg_subdirs", []) or []))
        all_sets.sort()  # deterministic CLI ordering
        cmd = [
            sys.executable, "-m", "chocolatechip.model_training.dataset_setup",
            "--root", str(root),
            "--sets", *all_sets,
            "--classes", str(spec.classes),
            "--names", spec.names,
            "--prefix", spec.prefix,
            "--val-frac", "0.20",
            "--seed", str(seed_val),
            "--exts", *[e.lower() for e in spec.exts],
        ]
        if spec.neg_subdirs:
            cmd += ["--neg-subdirs", *spec.neg_subdirs]
        if spec.legos:
            cmd += ["--legos"]

    _eprint("[split]", " ".join(cmd))
    subprocess.check_call(cmd)
