<!-- markdownlint-disable -->

# API Overview

## Modules

- [`base`](./base.md#module-base): Base classes and utilities for conformal prediction.
- [`base.cp`](./base.cp.md#module-basecp): Base module for conformal prediction classes.
- [`base.data`](./base.data.md#module-basedata): Base data structures for predictions, parameters, and conformalized results.
- [`base.loss`](./base.loss.md#module-baseloss): Loss functions and non-conformity scores for conformal prediction.
- [`base.models`](./base.models.md#module-basemodels): Base model class for machine learning models in the cods library.
- [`base.optim`](./base.optim.md#module-baseoptim): Optimizers for conformal prediction and related search procedures.
- [`base.tr`](./base.tr.md#module-basetr): Base training and tolerance region utilities for conformal prediction.
- [`classif`](./classif.md#module-classif): Conformal classification module.
- [`classif.cp`](./classif.cp.md#module-classifcp): Conformalizer for conformal classification tasks.
- [`classif.data`](./classif.data.md#module-classifdata): Data handling for classification tasks.
- [`classif.data.datasets`](./classif.data.datasets.md#module-classifdatadatasets): Datasets for conformal classification tasks.
- [`classif.data.predictions`](./classif.data.predictions.md#module-classifdatapredictions): Data structures for classification predictions.
- [`classif.loss`](./classif.loss.md#module-classifloss): Loss functions for conformal classification.
- [`classif.metrics`](./classif.metrics.md#module-classifmetrics): Metrics for evaluating conformal classification predictions.
- [`classif.models`](./classif.models.md#module-classifmodels): Model wrapper for conformal classification tasks.
- [`classif.score`](./classif.score.md#module-classifscore): Non-conformity scores for conformal classification.
- [`classif.tr`](./classif.tr.md#module-classiftr): Tolerance region implementation for conformal classification.
- [`classif.visualization`](./classif.visualization.md#module-classifvisualization): Visualization utilities for conformal classification predictions.
- [`od`](./od.md#module-od): Object detection module for conformal prediction.
- [`od.cp`](./od.cp.md#module-odcp): Conformal prediction implementations for object detection.
- [`od.data`](./od.data.md#module-oddata): Data handling for object detection tasks.
- [`od.data.datasets`](./od.data.datasets.md#module-oddatadatasets): Dataset classes for object detection tasks.
- [`od.data.predictions`](./od.data.predictions.md#module-oddatapredictions): Data structures for object detection predictions and results.
- [`od.evaluate`](./od.evaluate.md#module-odevaluate): Evaluation functionality for object detection models with conformal prediction.
- [`od.loss`](./od.loss.md#module-odloss): Loss functions for object detection conformal prediction.
- [`od.metrics`](./od.metrics.md#module-odmetrics): Metrics computation and evaluation for object detection conformal prediction.
- [`od.models`](./od.models.md#module-odmodels): Object detection models for conformal prediction.
- [`od.models.detr`](./od.models.detr.md#module-odmodelsdetr): DETR (DEtection TRansformer) model implementation for object detection.
- [`od.models.model`](./od.models.model.md#module-odmodelsmodel): Base object detection model class for conformal prediction.
- [`od.models.utils`](./od.models.utils.md#module-odmodelsutils): Utility functions and classes for object detection models.
- [`od.models.yolo`](./od.models.yolo.md#module-odmodelsyolo): YOLO model implementation for object detection with conformal prediction.
- [`od.optim`](./od.optim.md#module-odoptim): Optimizers for conformal object detection calibration and risk control.
- [`od.score`](./od.score.md#module-odscore): Non-conformity scoring functions for object detection conformal prediction.
- [`od.utils`](./od.utils.md#module-odutils): Utility functions for object detection tasks and conformal prediction.
- [`od.visualization`](./od.visualization.md#module-odvisualization): Visualization utilities for conformal object detection predictions.

## Classes

- [`cp.Conformalizer`](./base.cp.md#class-conformalizer): Abstract base class for conformal prediction methods.
- [`data.ConformalizedPredictions`](./base.data.md#class-conformalizedpredictions): Abstract base class for conformalized prediction results.
- [`data.Parameters`](./base.data.md#class-parameters): Abstract base class for parameters.
- [`data.Predictions`](./base.data.md#class-predictions): Abstract base class for predictions.
- [`data.Results`](./base.data.md#class-results): Abstract base class for results.
- [`loss.Loss`](./base.loss.md#class-loss): Abstract base class for loss functions in conformal prediction.
- [`loss.NCScore`](./base.loss.md#class-ncscore): Abstract base class for non-conformity scores in conformal prediction.
- [`models.Model`](./base.models.md#class-model): Abstract base class for models in the cods library.
- [`optim.BinarySearchOptimizer`](./base.optim.md#class-binarysearchoptimizer): Optimizer using binary search in 1D (or multi-D) for risk calibration.
- [`optim.GaussianProcessOptimizer`](./base.optim.md#class-gaussianprocessoptimizer): Optimizer using Gaussian Process Regression (Bayesian optimization) for risk calibration.
- [`optim.MonteCarloOptimizer`](./base.optim.md#class-montecarlooptimizer): Optimizer using Monte Carlo random search for risk calibration.
- [`optim.Optimizer`](./base.optim.md#class-optimizer): Abstract base class for optimizers used in conformal prediction calibration.
- [`tr.CombiningToleranceRegions`](./base.tr.md#class-combiningtoleranceregions): Combine multiple tolerance regions, e.g., using Bonferroni correction.
- [`tr.ToleranceRegion`](./base.tr.md#class-toleranceregion): Abstract base class for tolerance region conformal predictors.
- [`cp.ClassificationConformalizer`](./classif.cp.md#class-classificationconformalizer): Implements conformal prediction for classification using various non-conformity scores.
- [`datasets.ClassificationDataset`](./classif.data.datasets.md#class-classificationdataset): Dataset for classification tasks with class index mapping and optional transforms.
- [`datasets.ImageNetDataset`](./classif.data.datasets.md#class-imagenetdataset): Dataset for ImageNet with automatic class index mapping and default transforms.
- [`predictions.ClassificationPredictions`](./classif.data.predictions.md#class-classificationpredictions): Container for predictions from a classification model.
- [`loss.ClassificationLoss`](./classif.loss.md#class-classificationloss): Abstract base class for classification loss functions.
- [`loss.LACLoss`](./classif.loss.md#class-lacloss): Loss function for Least Ambiguous Conformal (LAC) prediction sets.
- [`models.ClassificationModel`](./classif.models.md#class-classificationmodel): Model wrapper for classification tasks with prediction saving/loading.
- [`score.APSNCScore`](./classif.score.md#class-apsncscore): Non-conformity score for Adaptive Prediction Sets (APS).
- [`score.ClassifNCScore`](./classif.score.md#class-classifncscore): Abstract base class for classification non-conformity score functions.
- [`score.LACNCScore`](./classif.score.md#class-lacncscore): Non-conformity score for Least Ambiguous Conformal (LAC) prediction sets.
- [`tr.ClassificationToleranceRegion`](./classif.tr.md#class-classificationtoleranceregion): Tolerance region for conformal classification tasks.
- [`cp.AsymptoticLocalizationObjectnessConformalizer`](./od.cp.md#class-asymptoticlocalizationobjectnessconformalizer): A class that performs risk conformalization for object detection predictions with asymptotic localization and objectness.
- [`cp.ConfidenceConformalizer`](./od.cp.md#class-confidenceconformalizer): Conformalizer for confidence/objectness in object detection.
- [`cp.LocalizationConformalizer`](./od.cp.md#class-localizationconformalizer): A class for performing localization conformalization. Should be used within an ODConformalizer.
- [`cp.ODClassificationConformalizer`](./od.cp.md#class-odclassificationconformalizer): Conformalizer for classification in object detection tasks.
- [`cp.ODConformalizer`](./od.cp.md#class-odconformalizer): Class representing conformalizers for object detection tasks.
- [`datasets.MSCOCODataset`](./od.data.datasets.md#class-mscocodataset)
- [`datasets.VOCDataset`](./od.data.datasets.md#class-vocdataset)
- [`predictions.ODConformalizedPredictions`](./od.data.predictions.md#class-odconformalizedpredictions): Class representing conformalized predictions for object detection tasks.
- [`predictions.ODParameters`](./od.data.predictions.md#class-odparameters): Class representing parameters for object detection tasks.
- [`predictions.ODPredictions`](./od.data.predictions.md#class-odpredictions): Class representing predictions for object detection tasks.
- [`predictions.ODResults`](./od.data.predictions.md#class-odresults): Class representing results for object detection tasks.
- [`evaluate.Benchmark`](./od.evaluate.md#class-benchmark)
- [`loss.BoxCountRecallConfidenceLoss`](./od.loss.md#class-boxcountrecallconfidenceloss): Confidence loss based on the recall of box counts.
- [`loss.BoxCountThresholdConfidenceLoss`](./od.loss.md#class-boxcountthresholdconfidenceloss): Confidence loss based on whether the count of conformalized boxes meets or exceeds the count of true boxes.
- [`loss.BoxCountTwosidedConfidenceLoss`](./od.loss.md#class-boxcounttwosidedconfidenceloss): Confidence loss based on whether the absolute difference between true and predicted box counts exceeds a threshold.
- [`loss.BoxWiseIoULoss`](./od.loss.md#class-boxwiseiouloss): Box-wise IoU loss.
- [`loss.BoxWisePrecisionLoss`](./od.loss.md#class-boxwiseprecisionloss): Box-wise precision loss.
- [`loss.BoxWiseRecallLoss`](./od.loss.md#class-boxwiserecallloss): Box-wise recall loss: 1 - mean(areas of the union of the boxes).
- [`loss.ClassBoxWiseRecallLoss`](./od.loss.md#class-classboxwiserecallloss): A combined recall loss for both localization (box-wise recall) and classification.
- [`loss.ClassificationLossWrapper`](./od.loss.md#class-classificationlosswrapper): Wraps a standard classification loss for use in object detection.
- [`loss.NumberPredictionsGapLoss`](./od.loss.md#class-numberpredictionsgaploss): Loss based on the normalized difference between the number of true boxes and conformalized boxes.
- [`loss.ODBinaryClassificationLoss`](./od.loss.md#class-odbinaryclassificationloss): Binary classification loss for object detection.
- [`loss.ODLoss`](./od.loss.md#class-odloss): Base class for Object Detection losses.
- [`loss.PixelWiseRecallLoss`](./od.loss.md#class-pixelwiserecallloss): Pixel-wise recall loss.
- [`loss.ThresholdedBoxDistanceConfidenceLoss`](./od.loss.md#class-thresholdedboxdistanceconfidenceloss): Confidence loss based on a thresholded distance between true and predicted boxes.
- [`loss.ThresholdedRecallLoss`](./od.loss.md#class-thresholdedrecallloss): A recall loss that is 1 if the miscoverage (1 - recall) exceeds a threshold `beta`, and 0 otherwise.
- [`metrics.ODEvaluator`](./od.metrics.md#class-odevaluator): Evaluator for object detection predictions using specified loss functions.
- [`detr.DETRModel`](./od.models.detr.md#class-detrmodel)
- [`model.ODModel`](./od.models.model.md#class-odmodel)
- [`utils.ResizeChannels`](./od.models.utils.md#class-resizechannels)
- [`yolo.AlteredYOLO`](./od.models.yolo.md#class-alteredyolo): YOLO model wrapper with hooks to capture raw outputs and input shapes during prediction.
- [`yolo.YOLOModel`](./od.models.yolo.md#class-yolomodel): Object Detection model wrapper for YOLO with custom preprocessing and postprocessing.
- [`optim.FirstStepMonotonizingOptimizer`](./od.optim.md#class-firststepmonotonizingoptimizer): Optimizer for the first step of monotonic risk control in object detection.
- [`optim.SecondStepMonotonizingOptimizer`](./od.optim.md#class-secondstepmonotonizingoptimizer): Optimizer for the second step of monotonic risk control in object detection.
- [`score.MinAdditiveSignedAssymetricHausdorffNCScore`](./od.score.md#class-minadditivesignedassymetrichausdorffncscore): MinAdditiveSignedAssymetricHausdorffNCScore is a class that calculates the score using the minimum additive signed asymmetric Hausdorff distance.
- [`score.MinMultiplicativeSignedAssymetricHausdorffNCScore`](./od.score.md#class-minmultiplicativesignedassymetrichausdorffncscore): MinMultiplicativeSignedAssymetricHausdorffNCScore is a class that calculates the score using the minimum multiplicative signed asymmetric Hausdorff distance.
- [`score.ODNCScore`](./od.score.md#class-odncscore): ODNCScore is an abstract class for calculating the score in object detection tasks.
- [`score.ObjectnessNCScore`](./od.score.md#class-objectnessncscore): ObjectnessNCScore is a class that calculates the score for objectness prediction.
- [`score.UnionAdditiveSignedAssymetricHausdorffNCScore`](./od.score.md#class-unionadditivesignedassymetrichausdorffncscore): UnionAdditiveSignedAssymetricHausdorffNCScore is a class that calculates the score using the union additive signed asymmetric Hausdorff distance.
- [`score.UnionMultiplicativeSignedAssymetricHausdorffNCScore`](./od.score.md#class-unionmultiplicativesignedassymetrichausdorffncscore): UnionMultiplicativeSignedAssymetricHausdorffNCScore is a class that calculates the score using the union multiplicative signed asymmetric Hausdorff distance.

## Functions

- [`tr.bernstein`](./base.tr.md#function-bernstein): Bernstein's inequality bound for binomial proportion confidence intervals.
- [`tr.bernstein_emp`](./base.tr.md#function-bernstein_emp): Empirical Bernstein bound for binomial proportion confidence intervals.
- [`tr.bernstein_uni`](./base.tr.md#function-bernstein_uni): Uniform Bernstein bound for binomial proportion confidence intervals.
- [`tr.bernstein_uni_lim`](./base.tr.md#function-bernstein_uni_lim): Limited uniform Bernstein bound for binomial proportion confidence intervals.
- [`tr.binom_inv_cdf`](./base.tr.md#function-binom_inv_cdf): Inverse binomial CDF for confidence interval calculation.
- [`tr.hoeffding`](./base.tr.md#function-hoeffding): Hoeffding's inequality bound for binomial proportion confidence intervals.
- [`metrics.get_coverage`](./classif.metrics.md#function-get_coverage): Compute the coverage of the conformal prediction set.
- [`metrics.get_empirical_risk`](./classif.metrics.md#function-get_empirical_risk): Compute the empirical risk of the conformal prediction set.
- [`metrics.get_empirical_safety`](./classif.metrics.md#function-get_empirical_safety): Compute the empirical safety of the conformal prediction set.
- [`visualization.plot_predictions`](./classif.visualization.md#function-plot_predictions): Plot classification predictions and optionally conformalized sets for given indices.
- [`evaluate.parse_args`](./od.evaluate.md#function-parse_args)
- [`metrics.compute_global_coverage`](./od.metrics.md#function-compute_global_coverage): Compute the global coverage for object detection predictions.
- [`metrics.getAveragePrecision`](./od.metrics.md#function-getaverageprecision): Get the average precision for object detection predictions.
- [`metrics.getStretch`](./od.metrics.md#function-getstretch): Get the stretch of object detection predictions.
- [`metrics.get_recall_precision`](./od.metrics.md#function-get_recall_precision): Get the recall and precision for object detection predictions.
- [`metrics.plot_recall_precision`](./od.metrics.md#function-plot_recall_precision): Plot the recall and precision given objectness threshold or IoU threshold.
- [`metrics.unroll_metrics`](./od.metrics.md#function-unroll_metrics): Compute and return various metrics for object detection predictions and conformalized predictions.
- [`detr.box_cxcywh_to_xyxy`](./od.models.detr.md#function-box_cxcywh_to_xyxy)
- [`detr.box_xyxy_to_cxcywh`](./od.models.detr.md#function-box_xyxy_to_cxcywh)
- [`detr.rescale_bboxes`](./od.models.detr.md#function-rescale_bboxes)
- [`utils.bayesod`](./od.models.utils.md#function-bayesod): _summary_.
- [`utils.filter_preds`](./od.models.utils.md#function-filter_preds)
- [`yolo.xywh2xyxy_scaled`](./od.models.yolo.md#function-xywh2xyxy_scaled): Convert bounding boxes from center (x, y, w, h) format to (x0, y0, x1, y1) format and scale.
- [`utils.apply_margins`](./od.utils.md#function-apply_margins): Apply margins to predicted bounding boxes for conformal prediction.
- [`utils.assymetric_hausdorff_distance`](./od.utils.md#function-assymetric_hausdorff_distance): Calculate asymmetric Hausdorff distance between sets of boxes.
- [`utils.assymetric_hausdorff_distance_old`](./od.utils.md#function-assymetric_hausdorff_distance_old): Calculate asymmetric Hausdorff distance between true and predicted box (legacy version).
- [`utils.compute_risk_image_level`](./od.utils.md#function-compute_risk_image_level): Compute image-level risk for conformal prediction.
- [`utils.compute_risk_image_level_confidence`](./od.utils.md#function-compute_risk_image_level_confidence): Compute image-level confidence risk for conformal prediction.
- [`utils.compute_risk_object_level`](./od.utils.md#function-compute_risk_object_level): Input : conformal and true boxes of a all images.
- [`utils.contained`](./od.utils.md#function-contained): Compute the intersection over union (IoU) between two bounding boxes.
- [`utils.contained_old`](./od.utils.md#function-contained_old): Compute the intersection over union (IoU) between two bounding boxes.
- [`utils.f_iou`](./od.utils.md#function-f_iou): Compute the intersection over union (IoU) between two bounding boxes.
- [`utils.f_lac`](./od.utils.md#function-f_lac): Calculate LAC (Loss Adaptive Conformal) score for classification.
- [`utils.fast_covered_areas_of_gt`](./od.utils.md#function-fast_covered_areas_of_gt)
- [`utils.generalized_iou`](./od.utils.md#function-generalized_iou): Compute the Generalized Intersection over Union (GIoU) between two bounding boxes.
- [`utils.get_covered_areas_of_gt_union`](./od.utils.md#function-get_covered_areas_of_gt_union)
- [`utils.match_predictions_to_true_boxes`](./od.utils.md#function-match_predictions_to_true_boxes): Match predictions to true boxes. Done in place, modifies the preds object.
- [`utils.mesh_func`](./od.utils.md#function-mesh_func): Compute mesh function.
- [`utils.rank_distance`](./od.utils.md#function-rank_distance): Calculate rank distance between true and predicted classes.
- [`utils.vectorized_generalized_iou`](./od.utils.md#function-vectorized_generalized_iou): Compute the Generalized Intersection over Union (GIoU) between two sets of
- [`visualization.create_pdf_with_plots`](./od.visualization.md#function-create_pdf_with_plots): Create a PDF with plots for each image in the predictions.
- [`visualization.plot_histograms_predictions`](./od.visualization.md#function-plot_histograms_predictions): Plot histograms of true boxes, predicted boxes, and thresholded predictions.
- [`visualization.plot_preds`](./od.visualization.md#function-plot_preds): Plot the predictions of an object detection model for a given image index.


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
