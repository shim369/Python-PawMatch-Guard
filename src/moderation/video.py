"""Video handling with ffmpeg: frame extraction (1 fps) and a representative thumbnail."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_available() -> bool:
    """True only if ffmpeg exists *and runs* (a broken install, e.g. a missing shared
    library, is treated as unavailable)."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        return subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    except OSError:
        return False


def extract_frames(video: Path, out_dir: Path, fps: float = 1.0,
                   max_frames: int = 120) -> list[Path]:
    """Extract ``fps`` frames per second as JPEGs; returns the sorted frame paths."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is not installed")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(video),
        "-vf", f"fps={fps}", "-frames:v", str(max_frames), "-q:v", "2",
        str(out_dir / "frame_%04d.jpg"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr.strip()[:300]}")
    return sorted(out_dir.glob("frame_*.jpg"))


def pick_thumbnail(frames: list[Path], risks: list[float]) -> Path:
    """Representative thumbnail: the *safest* frame (lowest risk), so the preview shown in
    the app is never the frame that triggered a warning."""
    if not frames or len(frames) != len(risks):
        raise ValueError("frames and risks must be non-empty and the same length")
    return frames[min(range(len(frames)), key=risks.__getitem__)]
