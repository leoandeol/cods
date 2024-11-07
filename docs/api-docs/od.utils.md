<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.utils`





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mesh_func`

```python
mesh_func(x1: int, y1: int, x2: int, y2: int, pbs: Tensor) → Tensor
```

Compute mesh function. 



**Args:**
 
---- 
 - <b>`x1`</b> (int):  x-coordinate of the top-left corner of the bounding box. 
 - <b>`y1`</b> (int):  y-coordinate of the top-left corner of the bounding box. 
 - <b>`x2`</b> (int):  x-coordinate of the bottom-right corner of the bounding box. 
 - <b>`y2`</b> (int):  y-coordinate of the bottom-right corner of the bounding box. 
 - <b>`pbs`</b> (List[List[int]]):  List of predicted bounding boxes. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Mesh function. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_covered_areas_of_gt_union`

```python
get_covered_areas_of_gt_union(pred_boxes, true_boxes)
```

Compute the covered areas of ground truth bounding boxes using union. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (List[List[int]]):  List of predicted bounding boxes. 
 - <b>`true_boxes`</b> (List[List[int]]):  List of ground truth bounding boxes. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Covered areas of ground truth bounding boxes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `contained`

```python
contained(tb, pb)
```

Compute the intersection over union (IoU) between two bounding boxes. 



**Args:**
 
---- 
 - <b>`tb`</b> (List[int]):  Ground truth bounding box. 
 - <b>`pb`</b> (List[int]):  Predicted bounding box. 



**Returns:**
 
------- 
 - <b>`float`</b>:  Intersection over union (IoU) value. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `f_iou`

```python
f_iou(boxA, boxB)
```

Compute the intersection over union (IoU) between two bounding boxes. 



**Args:**
 
---- 
 - <b>`boxA`</b> (List[int]):  First bounding box. 
 - <b>`boxB`</b> (List[int]):  Second bounding box. 



**Returns:**
 
------- 
 - <b>`float`</b>:  Intersection over union (IoU) value. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `assymetric_hausdorff_distance`

```python
assymetric_hausdorff_distance(true_box, pred_box)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `match_predictions_to_true_boxes`

```python
match_predictions_to_true_boxes(
    preds,
    distance_function,
    overload_confidence_threshold=None,
    verbose=False
) → None
```

Matching predictions to true boxes. Done in place, modifies the preds object. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `apply_margins`

```python
apply_margins(pred_boxes: List[Tensor], Qs, mode='additive')
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L256"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_risk_object_level`

```python
compute_risk_object_level(
    conformalized_predictions,
    predictions,
    loss,
    return_list: bool = False
) → Tensor
```

Input : conformal and true boxes of a all images 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_risk_image_level`

```python
compute_risk_image_level(
    conformalized_predictions,
    predictions,
    loss,
    return_list: bool = False
) → Tensor
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
