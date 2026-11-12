import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix as sk_cm

from src.metrics import accuracy, confusion_matrix, precision_recall_f1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    c = 7
    y_true = rng.integers(0, c, 500)
    y_pred = np.where(rng.random(500) < 0.6, y_true, rng.integers(0, c, 500))
    cm = confusion_matrix(y_true, y_pred, c)
    assert np.array_equal(cm, sk_cm(y_true, y_pred, labels=range(c)))
    assert accuracy(cm) == pytest.approx(accuracy_score(y_true, y_pred))
    prf = precision_recall_f1(cm)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(c), zero_division=0
    )
    assert np.allclose(prf.precision, p)
    assert np.allclose(prf.recall, r)
    assert np.allclose(prf.f1, f)
    assert np.array_equal(prf.support, s)
    assert prf.macro_f1 == pytest.approx(
        f1_score(y_true, y_pred, average="macro", labels=range(c), zero_division=0)
    )


def test_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 3]), np.array([0, 1]), 3)


def test_absent_class_is_zero() -> None:
    cm = confusion_matrix(np.array([0, 0]), np.array([0, 0]), 3)
    assert precision_recall_f1(cm).f1.tolist() == [1.0, 0.0, 0.0]
