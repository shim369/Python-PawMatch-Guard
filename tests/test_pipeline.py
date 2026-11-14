import numpy as np
import pytest

from src.moderation.pipeline import ModerationPipeline
from src.moderation.rules import Detection, Verdict


class FakeDetector:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, image):
        return list(self.dets)


class FakeClassifier:
    def __init__(self, species="cat", conf=0.99):
        self.out = (species, conf)

    def predict(self, image):
        return self.out


IMG = np.full((64, 64, 3), 100, np.uint8)
ANIMAL = Detection("animal", 0.95, (5, 5, 50, 50))


def test_clean_image_publishes_untouched() -> None:
    p = ModerationPipeline(FakeDetector([ANIMAL]), FakeClassifier())
    res, out = p.moderate_image(IMG, "cat")
    assert res.decision.verdict is Verdict.PUBLISH and out is IMG


def test_pii_is_blurred_in_published_image() -> None:
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    face = Detection("face", 0.9, (10, 10, 40, 40))
    p = ModerationPipeline(FakeDetector([ANIMAL, face]), FakeClassifier())
    res, out = p.moderate_image(img, "cat")
    assert res.decision.verdict is Verdict.PUBLISH
    assert out[15:35, 15:35].std() < img[15:35, 15:35].std() / 2


def test_mismatch_and_reject() -> None:
    assert ModerationPipeline(FakeDetector([ANIMAL]), FakeClassifier("dog")
                              ).moderate_image(IMG, "cat")[0].decision.verdict is Verdict.REVIEW
    assert ModerationPipeline(FakeDetector([]), FakeClassifier()
                              ).moderate_image(IMG, "cat")[0].decision.verdict is Verdict.REJECT


def test_video_worst_frame_wins_and_thumbnail_is_safe(tmp_path, monkeypatch) -> None:
    from PIL import Image

    import src.moderation.pipeline as mod

    frames = []
    for i in range(3):
        f = tmp_path / f"frame_{i}.jpg"
        Image.fromarray(np.full((32, 32, 3), 50 * (i + 1), np.uint8)).save(f)
        frames.append(f)
    monkeypatch.setattr(mod, "extract_frames", lambda *a, **k: frames)

    class PerFrameDetector:
        def __init__(self):
            self.n = 0

        def detect(self, image):
            self.n += 1
            if self.n == 2:  # second frame has no animal -> reject
                return []
            return [Detection("animal", 0.9 if self.n == 1 else 0.99, (0, 0, 10, 10))]

    p = ModerationPipeline(PerFrameDetector(), FakeClassifier())
    res = p.moderate_video(tmp_path / "v.mp4", tmp_path / "w", "cat")
    assert res.decision.verdict is Verdict.REJECT
    assert [d.verdict for d in res.frame_decisions] == [
        Verdict.PUBLISH, Verdict.REJECT, Verdict.PUBLISH]
    assert abs(float(res.thumbnail.mean()) - 150) < 5  # frame 3: safest (score 0.99)


def test_empty_video_raises(tmp_path, monkeypatch) -> None:
    import src.moderation.pipeline as mod

    monkeypatch.setattr(mod, "extract_frames", lambda *a, **k: [])
    with pytest.raises(RuntimeError):
        ModerationPipeline(FakeDetector([]), FakeClassifier()).moderate_video(
            tmp_path / "v.mp4", tmp_path, "cat")


def test_video_with_pii_is_never_auto_published(tmp_path, monkeypatch) -> None:
    from PIL import Image

    import src.moderation.pipeline as mod

    f = tmp_path / "frame_0.jpg"
    Image.fromarray(np.full((32, 32, 3), 90, np.uint8)).save(f)
    monkeypatch.setattr(mod, "extract_frames", lambda *a, **k: [f])
    dets = [ANIMAL, Detection("face", 0.95, (2, 2, 12, 12))]
    res = ModerationPipeline(FakeDetector(dets), FakeClassifier()).moderate_video(
        tmp_path / "v.mp4", tmp_path, "cat")
    assert res.decision.verdict is Verdict.REVIEW
    assert "video_pii_unblurred" in res.decision.reasons
