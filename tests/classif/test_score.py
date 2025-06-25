"""Tests for classification scoring functions."""
import numpy as np

from cods.classif.score import LACNCScore


def test_lac():
    """Test LAC scoring function."""
    Ypred = np.ones(4) * 0.25
    ytrue = 2
    lac = LACNCScore(4)
    score = lac(Ypred, ytrue)
    assert score == 0.75
