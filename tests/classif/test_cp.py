"""Tests for classification conformal prediction."""

from cods.classif.cp import ClassificationConformalizer


def test_classification_conformalizer_init():
    """Test ClassificationConformalizer initialization."""
    conf = ClassificationConformalizer(method="lac", preprocess="softmax")
    assert conf is not None
    assert conf.method == "lac"
    assert conf.preprocess == "softmax"


def test_classification_conformalizer_lac_method():
    """Test LAC method in ClassificationConformalizer."""
    conf = ClassificationConformalizer(method="lac", preprocess="softmax")
    assert conf.method == "lac"
    assert hasattr(conf, "ACCEPTED_METHODS")
    assert "lac" in conf.ACCEPTED_METHODS


def test_classification_conformalizer_aps_method():
    """Test APS method in ClassificationConformalizer."""
    conf = ClassificationConformalizer(method="aps", preprocess="softmax")
    assert conf is not None
    assert conf.method == "aps"


def test_classification_conformalizer_with_softmax_preprocess():
    """Test ClassificationConformalizer with softmax preprocessing."""
    conf_softmax = ClassificationConformalizer(
        method="lac", preprocess="softmax"
    )

    assert conf_softmax.preprocess == "softmax"
    assert hasattr(conf_softmax, "ACCEPTED_PREPROCESS")
    assert "softmax" in conf_softmax.ACCEPTED_PREPROCESS


def test_classification_conformalizer_device():
    """Test ClassificationConformalizer device setting."""
    conf_cpu = ClassificationConformalizer(method="lac", device="cpu")
    conf_cuda = ClassificationConformalizer(method="lac", device="cuda")

    assert conf_cpu.device == "cpu"
    assert conf_cuda.device == "cuda"
