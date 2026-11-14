from pathlib import Path

from src.moderation.rules import Decision, Detection, Prediction, Thresholds, Verdict, decide, worst

ANIMAL = Detection("animal", 0.95, (0, 0, 10, 10))


def P(*dets, species="cat", conf=0.99):
    return Prediction(tuple(dets), species, conf)


def test_rule1_no_animal_rejects() -> None:
    assert decide(P(), "cat").verdict is Verdict.REJECT
    assert decide(P(Detection("animal", 0.1, (0, 0, 1, 1))), "cat").reasons == ("no_animal",)
    # PII alone does not make a post acceptable.
    assert decide(P(Detection("face", 0.99, (0, 0, 5, 5))), "cat").verdict is Verdict.REJECT


def test_uncertain_animal_goes_to_review() -> None:
    d = decide(P(Detection("animal", 0.4, (0, 0, 1, 1))), "cat")
    assert d.verdict is Verdict.REVIEW and "uncertain_animal" in d.reasons


def test_rule2_pii_blur_and_publish() -> None:
    face = Detection("face", 0.9, (1, 1, 4, 4))
    d = decide(P(ANIMAL, face), "cat")
    assert d.verdict is Verdict.PUBLISH
    assert d.blur_boxes == ((1, 1, 4, 4),) and "pii_blurred" in d.reasons


def test_rule2_low_confidence_pii_needs_review_but_is_still_blurred() -> None:
    d = decide(P(ANIMAL, Detection("plate", 0.4, (2, 2, 6, 6))), "cat")
    assert d.verdict is Verdict.REVIEW and d.blur_boxes == ((2, 2, 6, 6),)
    # below the blur minimum it is ignored
    d2 = decide(P(ANIMAL, Detection("text", 0.1, (2, 2, 6, 6))), "cat")
    assert d2.verdict is Verdict.PUBLISH and d2.blur_boxes == ()


def test_rule3_species_mismatch() -> None:
    d = decide(P(ANIMAL, species="dog"), "cat")
    assert d.verdict is Verdict.REVIEW and d.reasons == ("species_mismatch",)
    assert decide(P(ANIMAL, species="dog"), None).verdict is Verdict.PUBLISH


def test_rule4_clean_post_publishes() -> None:
    d = decide(P(ANIMAL), "cat")
    assert d == Decision(Verdict.PUBLISH, (), (), d.risk)


def test_reject_beats_review_beats_publish_and_worst_frame_wins() -> None:
    ok = decide(P(ANIMAL), "cat")
    review = decide(P(ANIMAL, species="dog"), "cat")
    reject = decide(P(), "cat")
    assert worst([ok, review, reject]) is reject
    assert worst([ok, review]) is review
    assert worst([ok]) is ok


def test_thresholds_from_yaml() -> None:
    th = Thresholds.from_yaml(Path("configs/moderation.yaml"))
    assert th == Thresholds()
