from cods.od.cp import ODConformalizer


def test_conformalizer():
    conf = ODConformalizer(
        backend="auto",
        guarantee_level="image",
        matching_function="mix",
        multiple_testing_correction=None,
        confidence_method="box_count_recall",
        localization_method="pixelwise",
        localization_prediction_set="additive",
        classification_method="binary",
        classification_prediction_set="lac",
    )
