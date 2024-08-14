<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.utils`





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_classif_preds_from_od_preds`

```python
get_classif_preds_from_od_preds(
    preds: ODPredictions
) → ClassificationPredictions
```

Convert object detection predictions to classification predictions. 



**Args:**
 
 - <b>`preds`</b> (ODPredictions):  Object detection predictions. 



**Returns:**
 
 - <b>`ClassificationPredictions`</b>:  Classification predictions. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `flatten_conf_cls`

```python
flatten_conf_cls(conf_cls: List[List[Tensor]]) → List[Tensor]
```

Flatten nested arrays into a single list. 



**Args:**
 
 - <b>`conf_cls`</b> (List[List[torch.Tensor]]):  Nested arrays. 



**Returns:**
 
 - <b>`List[torch.Tensor]`</b>:  Flattened list. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_conf_cls_for_od`

```python
get_conf_cls_for_od(
    od_preds: ODPredictions,
    conformalizer: Union[ClassificationConformalizer, ClassificationToleranceRegion]
) → List[List[Tensor]]
```

Get confidence scores for object detection predictions. 



**Args:**
 
 - <b>`od_preds`</b> (ODPredictions):  Object detection predictions. 
 - <b>`conformalizer`</b> (Union[ClassificationConformalizer, ClassificationToleranceRegion]):  Conformalizer object. 



**Returns:**
 
 - <b>`List[List[torch.Tensor]]`</b>:  Confidence scores for each object detection prediction. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `evaluate_cls_conformalizer`

```python
evaluate_cls_conformalizer(
    od_preds: ODPredictions,
    conf_cls: List[List[Tensor]],
    conformalizer: Union[ClassificationConformalizer, ClassificationToleranceRegion],
    verbose: bool = False
) → Tensor
```

Evaluate the performance of a classification conformalizer. 



**Args:**
 
 - <b>`od_preds`</b> (ODPredictions):  Object detection predictions. 
 - <b>`conf_cls`</b> (List[List[torch.Tensor]]):  Confidence scores for each object detection prediction. 
 - <b>`conformalizer`</b> (Union[ClassificationConformalizer, ClassificationToleranceRegion]):  Conformalizer object. 
 - <b>`verbose`</b> (bool, optional):  Whether to print verbose output. Defaults to False. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Coverage and set size for each object detection prediction. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mesh_func`

```python
mesh_func(x1, y1, x2, y2, pbs)
```

Compute mesh function. 



**Args:**
 
 - <b>`x1`</b> (int):  x-coordinate of the top-left corner of the bounding box. 
 - <b>`y1`</b> (int):  y-coordinate of the top-left corner of the bounding box. 
 - <b>`x2`</b> (int):  x-coordinate of the bottom-right corner of the bounding box. 
 - <b>`y2`</b> (int):  y-coordinate of the bottom-right corner of the bounding box. 
 - <b>`pbs`</b> (List[List[int]]):  List of predicted bounding boxes. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Mesh function. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_covered_areas_of_gt_union`

```python
get_covered_areas_of_gt_union(pred_boxes, true_boxes)
```

Compute the covered areas of ground truth bounding boxes using union. 



**Args:**
 
 - <b>`pred_boxes`</b> (List[List[int]]):  List of predicted bounding boxes. 
 - <b>`true_boxes`</b> (List[List[int]]):  List of ground truth bounding boxes. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Covered areas of ground truth bounding boxes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L246"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_covered_areas_of_gt_max`

```python
get_covered_areas_of_gt_max(pred_boxes, true_boxes)
```

Compute the covered areas of ground truth bounding boxes using maximum. 



**Args:**
 
 - <b>`pred_boxes`</b> (List[List[int]]):  List of predicted bounding boxes. 
 - <b>`true_boxes`</b> (List[List[int]]):  List of ground truth bounding boxes. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Covered areas of ground truth bounding boxes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `contained`

```python
contained(tb, pb)
```

Compute the intersection over union (IoU) between two bounding boxes. 



**Args:**
 
 - <b>`tb`</b> (List[int]):  Ground truth bounding box. 
 - <b>`pb`</b> (List[int]):  Predicted bounding box. 



**Returns:**
 
 - <b>`float`</b>:  Intersection over union (IoU) value. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L297"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `f_iou`

```python
f_iou(boxA, boxB)
```

Compute the intersection over union (IoU) between two bounding boxes. 



**Args:**
 
 - <b>`boxA`</b> (List[int]):  First bounding box. 
 - <b>`boxB`</b> (List[int]):  Second bounding box. 



**Returns:**
 
 - <b>`float`</b>:  Intersection over union (IoU) value. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L321"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `matching_by_iou`

```python
matching_by_iou(preds, verbose=False)
```

Perform matching between ground truth and predicted bounding boxes based on IoU. 



**Args:**
 
 - <b>`preds`</b>:  Object detection predictions. 
 - <b>`verbose`</b> (bool, optional):  Whether to print verbose output. Defaults to False. 



**Returns:**
 
 - <b>`List[List[int]]`</b>:  Matching indices. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L379"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `apply_margins`

```python
apply_margins(pred_boxes, Qs, mode='additive')
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L399"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_risk_box_level`

```python
compute_risk_box_level(conf_boxes, true_boxes, loss)
```

Input : conformal and true boxes of a all images 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L415"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_risk_cls_box_level`

```python
compute_risk_cls_box_level(conf_boxes, conf_cls, true_boxes, true_cls, loss)
```

Input : conformal and true boxes of a all images 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L433"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_risk_cls_image_level`

```python
compute_risk_cls_image_level(conf_boxes, conf_cls, true_boxes, true_cls, loss)
```

Input : conformal and true boxes of a all images 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L450"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_risk_image_level`

```python
compute_risk_image_level(conf_boxes, true_boxes, loss)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
