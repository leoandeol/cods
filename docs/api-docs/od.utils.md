<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.utils`





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
 - <b>`pbs`</b> (torch.Tensor):  List of predicted bounding boxes. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Mesh function. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_covered_areas_of_gt_union`

```python
get_covered_areas_of_gt_union(pred_boxes, true_boxes)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fast_covered_areas_of_gt`

```python
fast_covered_areas_of_gt(pred_boxes, true_boxes)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `contained`

```python
contained(tb: Tensor, pb: Tensor) → Tensor
```

Compute the intersection over union (IoU) between two bounding boxes. 



**Args:**
 
---- 
 - <b>`tb`</b> (torch.Tensor):  Ground truth bounding boxes (N, 4). 
 - <b>`pb`</b> (torch.Tensor):  Predicted bounding boxes (N, 4). 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  IoU values (N,). 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `contained_old`

```python
contained_old(tb, pb)
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

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `generalized_iou`

```python
generalized_iou(boxA, boxB)
```

Compute the Generalized Intersection over Union (GIoU) between two bounding boxes. 



**Args:**
 
---- 
 - <b>`boxA`</b> (List[int]):  First bounding box. 
 - <b>`boxB`</b> (List[int]):  Second bounding box. 



**Returns:**
 
------- 
 - <b>`float`</b>:  Generalized Intersection over Union (GIoU) value. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L214"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vectorized_generalized_iou`

```python
vectorized_generalized_iou(boxesA: ndarray, boxesB: ndarray) → ndarray
```

Compute the Generalized Intersection over Union (GIoU) between two sets of bounding boxes. 

Calculates the GIoU for every pair of boxes between boxesA and boxesB. 



**Args:**
 
---- 
 - <b>`boxesA`</b> (np.ndarray):  A NumPy array of shape (N, 4) representing N bounding boxes.  Each row is [x1, y1, x2, y2]. 
 - <b>`boxesB`</b> (np.ndarray):  A NumPy array of shape (M, 4) representing M bounding boxes.  Each row is [x1, y1, x2, y2]. 



**Returns:**
 
------- 
 - <b>`np.ndarray`</b>:  A NumPy array of shape (N, M) containing the GIoU values for  each pair of boxes (boxesA[i], boxesB[j]). 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If input arrays do not have shape (N, 4) or (M, 4). 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L328"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `assymetric_hausdorff_distance_old`

```python
assymetric_hausdorff_distance_old(true_box, pred_box)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L343"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `assymetric_hausdorff_distance`

```python
assymetric_hausdorff_distance(true_boxes, pred_boxes)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L353"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `f_lac`

```python
f_lac(true_cls, pred_cls)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L361"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rank_distance`

```python
rank_distance(true_cls, pred_cls)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L369"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `match_predictions_to_true_boxes`

```python
match_predictions_to_true_boxes(
    preds,
    distance_function,
    overload_confidence_threshold=None,
    verbose=False,
    hungarian=False,
    idx=None,
    class_factor: float = 0.25
) → None
```

Matching predictions to true boxes. Done in place, modifies the preds object. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L499"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `apply_margins`

```python
apply_margins(pred_boxes: List[Tensor], Qs, mode='additive')
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L533"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_risk_object_level`

```python
compute_risk_object_level(
    conformalized_predictions,
    predictions,
    loss,
    return_list: bool = False
) → Tensor
```

Input : conformal and true boxes of a all images. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L595"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/utils.py#L676"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_risk_image_level_confidence`

```python
compute_risk_image_level_confidence(
    conformalized_predictions,
    predictions,
    confidence_loss,
    other_losses=None,
    return_list: bool = False
) → Tensor
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
