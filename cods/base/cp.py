class Conformalizer:
    def __init__(self):
        pass

    def calibrate(self, preds, alpha=0.1):
        raise NotImplementedError(
            "Conformalizer is an abstract class, must be instantiated on a given task."
        )

    def conformalize(self, preds):
        raise NotImplementedError(
            "Conformalizer is an abstract class, must be instantiated on a given task."
        )

    # evaluate ? plot residuals ? what others ?
    def evaluate(self, preds, verbose=True):
        raise NotImplementedError(
            "Conformalizer is an abstract class, must be instantiated on a given task."
        )


# TODO: deprecated
# class RiskConformalizer(Conformalizer):
#     def __init__(self):
#         pass

#     def calibrate(self, preds, alpha=0.1):
#         raise NotImplementedError(
#             "RiskConformalizer is an abstract class, must be instantiated on a given task."
#         )

#     def calibrate_multidim(self, preds, alpha=0.1):
#         raise NotImplementedError(
#             "RiskConformalizer is an abstract class, must be instantiated on a given task."
#         )

#     def conformalize(self, preds):
#         raise NotImplementedError(
#             "RiskConformalizer is an abstract class, must be instantiated on a given task."
#         )

#     # evaluate ? plot residuals ? what others ?
#     def evaluate(self, preds, verbose=True):
#         raise NotImplementedError(
#             "RiskConformalizer is an abstract class, must be instantiated on a given task."
#         )


# TODO: unimplemented add optimization of bonferroni on another dataset
# class CombiningConformalPredictionSets(Conformalizer):
#     def __init__(self, *conformalizers, mode="bonferroni"):
#         self.conformalizers = conformalizers
#         self.mode = mode

#     def calibrate(self, preds, alpha=0.1, parameters=None):
#         if parameters is None:
#             parameters = [{} for _ in range(len(self.conformalizers))]
#         if self.mode == "bonferroni":
#             return list(
#                 [
#                     conformalizer.calibrate(
#                         preds, alpha=alpha / len(self.conformalizers), **parameters[i]
#                     )
#                     for i, conformalizer in enumerate(self.conformalizers)
#                 ]
#             )

#     def conformalize(self, preds):
#         return list(
#             [conformalizer.conformalize(preds) for conformalizer in self.conformalizers]
#         )

#     def evaluate(self, preds, verbose=True):
#         return list(
#             [
#                 conformalizer.evaluate(preds, verbose=verbose)
#                 for conformalizer in self.conformalizers
#             ]
#         )
