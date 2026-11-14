import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.moderation.blur import blur_regions
from src.moderation.video import extract_frames, ffmpeg_available, pick_thumbnail


def test_blur_only_changes_the_box() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = blur_regions(img, [(20, 20, 60, 60)], pad=0.0)
    assert (out[:19] == img[:19]).all() and (out[:, 70:] == img[:, 70:]).all()
    assert not (out[20:60, 20:60] == img[20:60, 20:60]).all()
    assert out[20:60, 20:60].std() < img[20:60, 20:60].std() / 2  # detail is gone


def test_blur_clips_and_ignores_degenerate_boxes() -> None:
    img = np.full((30, 30, 3), 128, np.uint8)
    out = blur_regions(img, [(-10, -10, 8, 8), (50, 50, 60, 60), (5, 5, 5, 5)])
    assert out.shape == img.shape


def test_input_not_modified() -> None:
    img = np.random.default_rng(1).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    before = img.copy()
    blur_regions(img, [(5, 5, 30, 30)])
    assert (img == before).all()


def test_pick_thumbnail_is_safest_frame() -> None:
    frames = [Path(f"f{i}.jpg") for i in range(3)]
    assert pick_thumbnail(frames, [0.9, 0.1, 0.5]) == frames[1]
    with pytest.raises(ValueError):
        pick_thumbnail([], [])


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_extract_one_frame_per_second(tmp_path) -> None:
    video = tmp_path / "t.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc=duration=3:size=64x48:rate=10", "-pix_fmt", "yuv420p", str(video)],
        check=True)
    frames = extract_frames(video, tmp_path / "frames", fps=1.0)
    assert len(frames) == 3
    with pytest.raises(RuntimeError):
        extract_frames(tmp_path / "missing.mp4", tmp_path / "x")
