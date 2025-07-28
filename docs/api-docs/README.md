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
- [`od.models.utils`](./od.models.utils.md#module-odmodelsutils)
- [`od.models.yolo`](./od.models.yolo.md#module-odmodelsyolo)
- [`od.optim`](./od.optim.md#module-odoptim)
- [`od.score`](./od.score.md#module-odscore)
- [`od.utils`](./od.utils.md#module-odutils)
- [`od.visualization`](./od.visualization.md#module-odvisualization)

## Classes

- [`cp.Conformalizer`](./base.cp.md#class-conformalizer)
- [`data.ConformalizedPredictions`](./base.data.md#class-conformalizedpredictions): Abstract class for results
- [`data.Parameters`](./base.data.md#class-parameters): Abstract class for parameters
- [`data.Predictions`](./base.data.md#class-predictions): Abstract class for predictions
- [`data.Results`](./base.data.md#class-results): Abstract class for results
- [`loss.Loss`](./base.loss.md#class-loss)
- [`loss.NCScore`](./base.loss.md#class-ncscore)
- [`models.Model`](./base.models.md#class-model)
- [`optim.BinarySearchOptimizer`](./base.optim.md#class-binarysearchoptimizer)
- [`optim.GaussianProcessOptimizer`](./base.optim.md#class-gaussianprocessoptimizer)
- [`optim.MonteCarloOptimizer`](./base.optim.md#class-montecarlooptimizer)
- [`optim.Optimizer`](./base.optim.md#class-optimizer)
- [`tr.CombiningToleranceRegions`](./base.tr.md#class-combiningtoleranceregions)
- [`tr.ToleranceRegion`](./base.tr.md#class-toleranceregion)
- [`cp.AsymptoticLocalizationObjectnessConformalizer`](./od.cp.md#class-asymptoticlocalizationobjectnessconformalizer): A class that performs risk conformalization for object detection predictions with asymptotic localization and objectness.
- [`cp.ConfidenceConformalizer`](./od.cp.md#class-confidenceconformalizer)
- [`cp.LocalizationConformalizer`](./od.cp.md#class-localizationconformalizer): A class for performing localization conformalization. Should be used within an ODConformalizer.
- [`cp.ODClassificationConformalizer`](./od.cp.md#class-odclassificationconformalizer)
- [`cp.ODConformalizer`](./od.cp.md#class-odconformalizer): Class representing conformalizers for object detection tasks.
- [`datasets.MSCOCODataset`](./od.data.datasets.md#class-mscocodataset)
- [`datasets.VOCDataset`](./od.data.datasets.md#class-vocdataset)
- [`predictions.ODConformalizedPredictions`](./od.data.predictions.md#class-odconformalizedpredictions): Class representing conformalized predictions for object detection tasks.
- [`predictions.ODParameters`](./od.data.predictions.md#class-odparameters): Class representing parameters for object detection tasks.
- [`predictions.ODPredictions`](./od.data.predictions.md#class-odpredictions): Class representing predictions for object detection tasks.
- [`predictions.ODResults`](./od.data.predictions.md#class-odresults): Class representing results for object detection tasks.
- [`evaluate.Benchmark`](./od.evaluate.md#class-benchmark)
- [`loss.BoxCountRecallConfidenceLoss`](./od.loss.md#class-boxcountrecallconfidenceloss)
- [`loss.BoxCountThresholdConfidenceLoss`](./od.loss.md#class-boxcountthresholdconfidenceloss)
- [`loss.BoxCountTwosidedConfidenceLoss`](./od.loss.md#class-boxcounttwosidedconfidenceloss)
- [`loss.BoxWiseIoULoss`](./od.loss.md#class-boxwiseiouloss): Box-wise PRECISION loss: 1 - mean(areas of the union of the boxes),
- [`loss.BoxWisePrecisionLoss`](./od.loss.md#class-boxwiseprecisionloss): Box-wise PRECISION loss: 1 - mean(areas of the union of the boxes),
- [`loss.BoxWiseRecallLoss`](./od.loss.md#class-boxwiserecallloss): Box-wise recall loss: 1 - mean(areas of the union of the boxes),
- [`loss.ClassBoxWiseRecallLoss`](./od.loss.md#class-classboxwiserecallloss)
- [`loss.ClassificationLossWrapper`](./od.loss.md#class-classificationlosswrapper)
- [`loss.NumberPredictionsGapLoss`](./od.loss.md#class-numberpredictionsgaploss)
- [`loss.ODBinaryClassificationLoss`](./od.loss.md#class-odbinaryclassificationloss)
- [`loss.ODLoss`](./od.loss.md#class-odloss)
- [`loss.PixelWiseRecallLoss`](./od.loss.md#class-pixelwiserecallloss)
- [`loss.ThresholdedBoxDistanceConfidenceLoss`](./od.loss.md#class-thresholdedboxdistanceconfidenceloss)
- [`loss.ThresholdedRecallLoss`](./od.loss.md#class-thresholdedrecallloss)
- [`metrics.ODEvaluator`](./od.metrics.md#class-odevaluator)
- [`detr.DETRModel`](./od.models.detr.md#class-detrmodel)
- [`model.ODModel`](./od.models.model.md#class-odmodel)
- [`utils.ResizeChannels`](./od.models.utils.md#class-resizechannels)
- [`yolo.AlteredYOLO`](./od.models.yolo.md#class-alteredyolo)
- [`yolo.YOLOModel`](./od.models.yolo.md#class-yolomodel)
- [`optim.FirstStepMonotonizingOptimizer`](./od.optim.md#class-firststepmonotonizingoptimizer)
- [`optim.SecondStepMonotonizingOptimizer`](./od.optim.md#class-secondstepmonotonizingoptimizer)
- [`score.MinAdditiveSignedAssymetricHausdorffNCScore`](./od.score.md#class-minadditivesignedassymetrichausdorffncscore): MinAdditiveSignedAssymetricHausdorffNCScore is a class that calculates the score using the minimum additive signed asymmetric Hausdorff distance.
- [`score.MinMultiplicativeSignedAssymetricHausdorffNCScore`](./od.score.md#class-minmultiplicativesignedassymetrichausdorffncscore): MinMultiplicativeSignedAssymetricHausdorffNCScore is a class that calculates the score using the minimum multiplicative signed asymmetric Hausdorff distance.
- [`score.ODNCScore`](./od.score.md#class-odncscore): ODNCScore is an abstract class for calculating the score in object detection tasks.
- [`score.ObjectnessNCScore`](./od.score.md#class-objectnessncscore): ObjectnessNCScore is a class that calculates the score for objectness prediction.
- [`score.UnionAdditiveSignedAssymetricHausdorffNCScore`](./od.score.md#class-unionadditivesignedassymetrichausdorffncscore): UnionAdditiveSignedAssymetricHausdorffNCScore is a class that calculates the score using the union additive signed asymmetric Hausdorff distance.
- [`score.UnionMultiplicativeSignedAssymetricHausdorffNCScore`](./od.score.md#class-unionmultiplicativesignedassymetrichausdorffncscore): UnionMultiplicativeSignedAssymetricHausdorffNCScore is a class that calculates the score using the union multiplicative signed asymmetric Hausdorff distance.

## Functions

- [`tr.bernstein`](./base.tr.md#function-bernstein)
- [`tr.bernstein_emp`](./base.tr.md#function-bernstein_emp)
- [`tr.bernstein_uni`](./base.tr.md#function-bernstein_uni)
- [`tr.bernstein_uni_lim`](./base.tr.md#function-bernstein_uni_lim)
- [`tr.binom_inv_cdf`](./base.tr.md#function-binom_inv_cdf)
- [`tr.hoeffding`](./base.tr.md#function-hoeffding)
- [`evaluate.parse_args`](./od.evaluate.md#function-parse_args)
- [`metrics.compute_global_coverage`](./od.metrics.md#function-compute_global_coverage): Compute the global coverage for object detection predictions. BOXWISE/IMAGEWISE #TODO
- [`metrics.getAveragePrecision`](./od.metrics.md#function-getaverageprecision): Get the average precision for object detection predictions.
- [`metrics.getStretch`](./od.metrics.md#function-getstretch): Get the stretch of object detection predictions.
- [`metrics.get_recall_precision`](./od.metrics.md#function-get_recall_precision): Get the recall and precision for object detection predictions.
- [`metrics.plot_recall_precision`](./od.metrics.md#function-plot_recall_precision): Plot the recall and precision given objectness threshold or IoU threshold.
- [`metrics.unroll_metrics`](./od.metrics.md#function-unroll_metrics)
- [`detr.box_cxcywh_to_xyxy`](./od.models.detr.md#function-box_cxcywh_to_xyxy)
- [`detr.box_xyxy_to_cxcywh`](./od.models.detr.md#function-box_xyxy_to_cxcywh)
- [`detr.rescale_bboxes`](./od.models.detr.md#function-rescale_bboxes)
- [`utils.bayesod`](./od.models.utils.md#function-bayesod): _summary_
- [`utils.filter_preds`](./od.models.utils.md#function-filter_preds)
- [`yolo.xywh2xyxy_scaled`](./od.models.yolo.md#function-xywh2xyxy_scaled)
- [`utils.apply_margins`](./od.utils.md#function-apply_margins)
- [`utils.assymetric_hausdorff_distance`](./od.utils.md#function-assymetric_hausdorff_distance)
- [`utils.assymetric_hausdorff_distance_old`](./od.utils.md#function-assymetric_hausdorff_distance_old)
- [`utils.compute_risk_image_level`](./od.utils.md#function-compute_risk_image_level)
- [`utils.compute_risk_image_level_confidence`](./od.utils.md#function-compute_risk_image_level_confidence)
- [`utils.compute_risk_object_level`](./od.utils.md#function-compute_risk_object_level): Input : conformal and true boxes of a all images
- [`utils.contained`](./od.utils.md#function-contained): Compute the intersection over union (IoU) between two bounding boxes.
- [`utils.contained_old`](./od.utils.md#function-contained_old): Compute the intersection over union (IoU) between two bounding boxes.
- [`utils.f_iou`](./od.utils.md#function-f_iou): Compute the intersection over union (IoU) between two bounding boxes.
- [`utils.f_lac`](./od.utils.md#function-f_lac)
- [`utils.fast_covered_areas_of_gt`](./od.utils.md#function-fast_covered_areas_of_gt)
- [`utils.generalized_iou`](./od.utils.md#function-generalized_iou): Compute the Generalized Intersection over Union (GIoU) between two bounding boxes.
- [`utils.get_covered_areas_of_gt_union`](./od.utils.md#function-get_covered_areas_of_gt_union)
- [`utils.match_predictions_to_true_boxes`](./od.utils.md#function-match_predictions_to_true_boxes): Matching predictions to true boxes. Done in place, modifies the preds object.
- [`utils.mesh_func`](./od.utils.md#function-mesh_func): Compute mesh function.
- [`utils.rank_distance`](./od.utils.md#function-rank_distance)
- [`utils.vectorized_generalized_iou`](./od.utils.md#function-vectorized_generalized_iou): Compute the Generalized Intersection over Union (GIoU) between two sets of
- [`visualization.create_pdf_with_plots`](./od.visualization.md#function-create_pdf_with_plots): Create a PDF with plots for each image in the predictions.
- [`visualization.plot_histograms_predictions`](./od.visualization.md#function-plot_histograms_predictions)
- [`visualization.plot_preds`](./od.visualization.md#function-plot_preds): Plot the predictions of an object detection model.


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
