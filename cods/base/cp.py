class Conformalizer:
    def __init__(self):
        pass

    def calibrate(self, preds, alpha=0.1):
        raise NotImplementedError(
            "Conformalizer is an abstract class, must be instantiated on a given task.",
        )

    def conformalize(self, preds):
        raise NotImplementedError(
            "Conformalizer is an abstract class, must be instantiated on a given task.",
        )

    # evaluate ? plot residuals ? what others ?
    def evaluate(self, preds, verbose=True):
        raise NotImplementedError(
            "Conformalizer is an abstract class, must be instantiated on a given task.",
        )
