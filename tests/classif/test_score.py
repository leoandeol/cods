import pytest
import numpy as np
from cods.classif.score import LACNCScore


def test_lac():
    Ypred = np.ones(4) * 0.25
    ytrue = 2
    lac = LACNCScore(4)
    score = lac(Ypred, ytrue)
    assert score == 0.75
