<!-- markdownlint-disable -->

# API Overview

## Modules

- [`base`](./base.md#module-base)
- [`base.cp`](./base.cp.md#module-basecp): Base module for conformal prediction classes.
- [`base.optim`](./base.optim.md#module-baseoptim): Optimizers for conformal prediction and related search procedures.
- [`od`](./od.md#module-od)
- [`od.utils`](./od.utils.md#module-odutils)

## Classes

- [`cp.Conformalizer`](./base.cp.md#class-conformalizer): Abstract base class for conformal prediction methods.
- [`optim.BinarySearchOptimizer`](./base.optim.md#class-binarysearchoptimizer): Optimizer using binary search in 1D (or multi-D) for risk calibration.
- [`optim.GaussianProcessOptimizer`](./base.optim.md#class-gaussianprocessoptimizer): Optimizer using Gaussian Process Regression (Bayesian optimization) for risk calibration.
- [`optim.MonteCarloOptimizer`](./base.optim.md#class-montecarlooptimizer): Optimizer using Monte Carlo random search for risk calibration.
- [`optim.Optimizer`](./base.optim.md#class-optimizer): Abstract base class for optimizers used in conformal prediction calibration.

## Functions

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


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
