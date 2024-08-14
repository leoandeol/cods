<!-- markdownlint-disable -->

# API Overview

## Modules

- [`base`](./base.md#module-base)
- [`base.cp`](./base.cp.md#module-basecp)
- [`base.data`](./base.data.md#module-basedata)
- [`base.loss`](./base.loss.md#module-baseloss)
- [`base.models`](./base.models.md#module-basemodels)
- [`base.optim`](./base.optim.md#module-baseoptim)
- [`base.tr`](./base.tr.md#module-basetr)
- [`od`](./od.md#module-od)
- [`od.cp`](./od.cp.md#module-odcp)
- [`od.data`](./od.data.md#module-oddata)
- [`od.data.datasets`](./od.data.datasets.md#module-oddatadatasets)
- [`od.data.predictions`](./od.data.predictions.md#module-oddatapredictions)
- [`od.evaluate`](./od.evaluate.md#module-odevaluate)
- [`od.loss`](./od.loss.md#module-odloss)
- [`od.metrics`](./od.metrics.md#module-odmetrics)
- [`od.models`](./od.models.md#module-odmodels)
- [`od.models.detr`](./od.models.detr.md#module-odmodelsdetr)
- [`od.models.model`](./od.models.model.md#module-odmodelsmodel)
- [`od.score`](./od.score.md#module-odscore)
- [`od.tr`](./od.tr.md#module-odtr)
- [`od.utils`](./od.utils.md#module-odutils)
- [`od.visualization`](./od.visualization.md#module-odvisualization)

## Classes

- [`cp.CombiningConformalPredictionSets`](./base.cp.md#class-combiningconformalpredictionsets)
- [`cp.Conformalizer`](./base.cp.md#class-conformalizer)
- [`cp.RiskConformalizer`](./base.cp.md#class-riskconformalizer)
- [`data.ConformalizedPredictions`](./base.data.md#class-conformalizedpredictions): Abstract class for results
- [`data.Parameters`](./base.data.md#class-parameters): Abstract class for parameters
- [`data.Predictions`](./base.data.md#class-predictions): Abstract class for predictions
- [`data.Results`](./base.data.md#class-results): Abstract class for results
- [`loss.BonferroniMultiLoss`](./base.loss.md#class-bonferronimultiloss)
- [`loss.HMPMultiLoss`](./base.loss.md#class-hmpmultiloss)
- [`loss.Loss`](./base.loss.md#class-loss)
- [`loss.MultiLoss`](./base.loss.md#class-multiloss)
- [`loss.NCScore`](./base.loss.md#class-ncscore)
- [`models.Model`](./base.models.md#class-model)
- [`optim.BinarySearchOptimizer`](./base.optim.md#class-binarysearchoptimizer)
- [`optim.GaussianProcessOptimizer`](./base.optim.md#class-gaussianprocessoptimizer)
- [`optim.MonteCarloOptimizer`](./base.optim.md#class-montecarlooptimizer)
- [`optim.Optimizer`](./base.optim.md#class-optimizer)
- [`tr.CombiningToleranceRegions`](./base.tr.md#class-combiningtoleranceregions)
- [`tr.ToleranceRegion`](./base.tr.md#class-toleranceregion)
- [`cp.AsymptoticLocalizationObjectnessRiskConformalizer`](./od.cp.md#class-asymptoticlocalizationobjectnessriskconformalizer): A class that performs risk conformalization for object detection predictions with asymptotic localization and objectness.
- [`cp.ConfidenceConformalizer`](./od.cp.md#class-confidenceconformalizer): A class that performs risk conformalization for localization tasks.
- [`cp.LocalizationConformalizer`](./od.cp.md#class-localizationconformalizer)
- [`cp.LocalizationRiskConformalizer`](./od.cp.md#class-localizationriskconformalizer): A class that performs risk conformalization for localization tasks.
- [`cp.ODClassificationConformalizer`](./od.cp.md#class-odclassificationconformalizer)
- [`cp.ODConformalizer`](./od.cp.md#class-odconformalizer): Class representing conformalizers for object detection tasks.
- [`cp.SeqGlobalODRiskConformalizer`](./od.cp.md#class-seqglobalodriskconformalizer)
- [`datasets.MSCOCODataset`](./od.data.datasets.md#class-mscocodataset)
- [`predictions.ODConformalizedPredictions`](./od.data.predictions.md#class-odconformalizedpredictions): Class representing conformalized predictions for object detection tasks.
- [`predictions.ODParameters`](./od.data.predictions.md#class-odparameters): Class representing parameters for object detection tasks.
- [`predictions.ODPredictions`](./od.data.predictions.md#class-odpredictions): Class representing predictions for object detection tasks.
- [`predictions.ODResults`](./od.data.predictions.md#class-odresults): Class representing results for object detection tasks.
- [`evaluate.Benchmark`](./od.evaluate.md#class-benchmark)
- [`loss.BoxWiseRecallLoss`](./od.loss.md#class-boxwiserecallloss)
- [`loss.ClassBoxWiseRecallLoss`](./od.loss.md#class-classboxwiserecallloss)
- [`loss.ClassificationLossWrapper`](./od.loss.md#class-classificationlosswrapper)
- [`loss.HausdorffSignedDistanceLoss`](./od.loss.md#class-hausdorffsigneddistanceloss)
- [`loss.MaximumLoss`](./od.loss.md#class-maximumloss)
- [`loss.NumberPredictionsGapLoss`](./od.loss.md#class-numberpredictionsgaploss)
- [`loss.ODLoss`](./od.loss.md#class-odloss)
- [`loss.ObjectnessLoss`](./od.loss.md#class-objectnessloss)
- [`loss.PixelWiseRecallLoss`](./od.loss.md#class-pixelwiserecallloss)
- [`detr.DETRModel`](./od.models.detr.md#class-detrmodel)
- [`detr.ResizeChannels`](./od.models.detr.md#class-resizechannels)
- [`model.ODModel`](./od.models.model.md#class-odmodel)
- [`score.MinAdditiveSignedAssymetricHausdorffNCScore`](./od.score.md#class-minadditivesignedassymetrichausdorffncscore): MinAdditiveSignedAssymetricHausdorffNCScore is a class that calculates the score using the minimum additive signed asymmetric Hausdorff distance.
- [`score.MinMultiplicativeSignedAssymetricHausdorffNCScore`](./od.score.md#class-minmultiplicativesignedassymetrichausdorffncscore): MinMultiplicativeSignedAssymetricHausdorffNCScore is a class that calculates the score using the minimum multiplicative signed asymmetric Hausdorff distance.
- [`score.ODNCScore`](./od.score.md#class-odncscore): ODNCScore is an abstract class for calculating the score in object detection tasks.
- [`score.ObjectnessNCScore`](./od.score.md#class-objectnessncscore): ObjectnessNCScore is a class that calculates the score for objectness prediction.
- [`score.UnionAdditiveSignedAssymetricHausdorffNCScore`](./od.score.md#class-unionadditivesignedassymetrichausdorffncscore): UnionAdditiveSignedAssymetricHausdorffNCScore is a class that calculates the score using the union additive signed asymmetric Hausdorff distance.
- [`score.UnionMultiplicativeSignedAssymetricHausdorffNCScore`](./od.score.md#class-unionmultiplicativesignedassymetrichausdorffncscore): UnionMultiplicativeSignedAssymetricHausdorffNCScore is a class that calculates the score using the union multiplicative signed asymmetric Hausdorff distance.
- [`tr.ConfidenceToleranceRegion`](./od.tr.md#class-confidencetoleranceregion): Tolerance region for object confidence tasks.
- [`tr.LocalizationToleranceRegion`](./od.tr.md#class-localizationtoleranceregion): Tolerance region for object localization tasks.
- [`tr.ODToleranceRegion`](./od.tr.md#class-odtoleranceregion)

## Functions

- [`tr.bernstein`](./base.tr.md#function-bernstein)
- [`tr.bernstein_emp`](./base.tr.md#function-bernstein_emp)
- [`tr.bernstein_uni`](./base.tr.md#function-bernstein_uni)
- [`tr.bernstein_uni_lim`](./base.tr.md#function-bernstein_uni_lim)
- [`tr.binom_inv_cdf`](./base.tr.md#function-binom_inv_cdf)
- [`tr.hoeffding`](./base.tr.md#function-hoeffding)
- [`metrics.compute_global_coverage`](./od.metrics.md#function-compute_global_coverage): Compute the global coverage for object detection predictions. BOXWISE/IMAGEWISE #TODO
- [`metrics.getAveragePrecision`](./od.metrics.md#function-getaverageprecision): Get the average precision for object detection predictions.
- [`metrics.getStretch`](./od.metrics.md#function-getstretch): Get the stretch of object detection predictions.
- [`metrics.get_recall_precision`](./od.metrics.md#function-get_recall_precision): Get the recall and precision for object detection predictions.
- [`metrics.plot_recall_precision`](./od.metrics.md#function-plot_recall_precision): Plot the recall and precision given objectness threshold or IoU threshold.
- [`metrics.unroll_metrics`](./od.metrics.md#function-unroll_metrics): Unroll the metrics for object detection predictions.
- [`detr.box_cxcywh_to_xyxy`](./od.models.detr.md#function-box_cxcywh_to_xyxy)
- [`detr.box_xyxy_to_cxcywh`](./od.models.detr.md#function-box_xyxy_to_cxcywh)
- [`detr.rescale_bboxes`](./od.models.detr.md#function-rescale_bboxes)
- [`utils.apply_margins`](./od.utils.md#function-apply_margins)
- [`utils.compute_risk_box_level`](./od.utils.md#function-compute_risk_box_level): Input : conformal and true boxes of a all images
- [`utils.compute_risk_cls_box_level`](./od.utils.md#function-compute_risk_cls_box_level): Input : conformal and true boxes of a all images
- [`utils.compute_risk_cls_image_level`](./od.utils.md#function-compute_risk_cls_image_level): Input : conformal and true boxes of a all images
- [`utils.compute_risk_image_level`](./od.utils.md#function-compute_risk_image_level)
- [`utils.contained`](./od.utils.md#function-contained): Compute the intersection over union (IoU) between two bounding boxes.
- [`utils.evaluate_cls_conformalizer`](./od.utils.md#function-evaluate_cls_conformalizer): Evaluate the performance of a classification conformalizer.
- [`utils.f_iou`](./od.utils.md#function-f_iou): Compute the intersection over union (IoU) between two bounding boxes.
- [`utils.flatten_conf_cls`](./od.utils.md#function-flatten_conf_cls): Flatten nested arrays into a single list.
- [`utils.get_classif_preds_from_od_preds`](./od.utils.md#function-get_classif_preds_from_od_preds): Convert object detection predictions to classification predictions.
- [`utils.get_conf_cls_for_od`](./od.utils.md#function-get_conf_cls_for_od): Get confidence scores for object detection predictions.
- [`utils.get_covered_areas_of_gt_max`](./od.utils.md#function-get_covered_areas_of_gt_max): Compute the covered areas of ground truth bounding boxes using maximum.
- [`utils.get_covered_areas_of_gt_union`](./od.utils.md#function-get_covered_areas_of_gt_union): Compute the covered areas of ground truth bounding boxes using union.
- [`utils.matching_by_iou`](./od.utils.md#function-matching_by_iou): Perform matching between ground truth and predicted bounding boxes based on IoU.
- [`utils.mesh_func`](./od.utils.md#function-mesh_func): Compute mesh function.
- [`visualization.plot_preds`](./od.visualization.md#function-plot_preds): Plot the predictions of an object detection model.


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
