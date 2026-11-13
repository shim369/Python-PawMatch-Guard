import numpy as np

from src.quality.agreement import box_agreement
from src.quality.annotation_checks import check_boxes, check_mask_box_consistency
from src.quality.noisy_labels import (
    detect_noisy,
    evaluate_detection,
    inject_label_noise,
    label_loss,
)


def test_check_boxes_flags_each_defect() -> None:
    boxes = np.array([[0, 0, 50, 50], [-5, 0, 20, 20], [10, 10, 10, 30], [0, 0, 20, 200]])
    issues = check_boxes(boxes, np.array([0, 1, 2, 9]), 100, 100, num_classes=4)
    codes = sorted((i.code, i.index) for i in issues)
    assert codes == [("bad_class", 3), ("out_of_bounds", 1), ("out_of_bounds", 3),
                     ("zero_area", 2)]
    assert check_boxes(boxes[:1], np.array([0]), 100, 100, 4) == []


def test_mask_box_consistency() -> None:
    mask = np.zeros((100, 100), np.uint8)
    mask[20:60, 30:70] = 1
    assert check_mask_box_consistency(mask, np.array([30, 20, 70, 60]), 100, 100) == []
    far = check_mask_box_consistency(mask, np.array([0, 0, 10, 10]), 100, 100)
    assert far[0].code == "mask_box_mismatch"
    assert check_mask_box_consistency(mask, np.array([30, 20, 70, 60]), 90, 100)[0].code == (
        "mask_shape"
    )
    mask[0, 0] = 5
    assert check_mask_box_consistency(mask, np.array([30, 20, 70, 60]), 100, 100)[0].code == (
        "mask_values"
    )


def test_agreement() -> None:
    r1 = [np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10], [50, 50, 60, 60]])]
    r2 = [np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10]])]
    a = box_agreement(r1, r2)
    assert a.matched == 2 and a.unmatched_a == 1 and a.mean_iou == 1.0
    assert abs(a.match_rate - 0.8) < 1e-9


def test_noise_injection_and_detection() -> None:
    rng = np.random.default_rng(0)
    n, c = 1000, 10
    labels = rng.integers(0, c, n)
    noisy, flipped = inject_label_noise(labels, c, 0.05)
    assert len(flipped) == 50 and (noisy[flipped] != labels[flipped]).all()
    assert (np.delete(noisy, flipped) == np.delete(labels, flipped)).all()
    # An accurate model's probabilities are peaked on the true label.
    probs = np.full((n, c), 0.02)
    probs[np.arange(n), labels] = 0.82
    probs /= probs.sum(1, keepdims=True)
    assert label_loss(probs, noisy)[flipped].min() > label_loss(probs, noisy).mean()
    rep = evaluate_detection(detect_noisy(probs, noisy, 0.05), flipped)
    assert rep.recall == 1.0 and rep.precision == 1.0
    dis = evaluate_detection(detect_noisy(probs, noisy, method="disagree"), flipped)
    assert dis.recall == 1.0
