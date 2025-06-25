<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/metrics.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.metrics`





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/metrics.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_global_coverage`

```python
compute_global_coverage(
    predictions: ODPredictions,
    parameters: ODParameters,
    conformalized_predictions: ODConformalizedPredictions,
    guarantee_level: str = 'object',
    confidence: bool = True,
    cls: bool = True,
    localization: bool = True,
    loss: Optional[Callable] = None
) → Tensor
```

Compute the global coverage for object detection predictions. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`parameters`</b> (ODParameters):  Parameters for object detection. 
 - <b>`conformalized_predictions`</b> (ODConformalizedPredictions):  Conformalized object detection predictions. 
 - <b>`guarantee_level`</b> (str, optional):  Level of coverage guarantee (e.g., 'object'). Defaults to 'object'. 
 - <b>`confidence`</b> (bool, optional):  Whether to consider confidence coverage. Defaults to True. 
 - <b>`cls`</b> (bool, optional):  Whether to consider class coverage. Defaults to True. 
 - <b>`localization`</b> (bool, optional):  Whether to consider localization coverage. Defaults to True. 
 - <b>`loss`</b> (Callable, optional):  Loss function to use. Defaults to None. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Global coverage tensor. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/metrics.py#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `getStretch`

```python
getStretch(od_predictions: ODPredictions, conf_boxes: list) → Tensor
```

Get the stretch of object detection predictions. 



**Args:**
 
---- 
 - <b>`od_predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`conf_boxes`</b> (list):  List of confidence boxes. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Stretch tensor. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/metrics.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_recall_precision`

```python
get_recall_precision(
    od_predictions: ODPredictions,
    IOU_THRESHOLD=0.5,
    SCORE_THRESHOLD=0.5,
    verbose=True,
    replace_iou=None
) → tuple
```

Get the recall and precision for object detection predictions. 



**Args:**
 
---- 
 - <b>`od_predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`IOU_THRESHOLD`</b> (float, optional):  IoU threshold. Defaults to 0.5. 
 - <b>`SCORE_THRESHOLD`</b> (float, optional):  Score threshold. Defaults to 0.5. 
 - <b>`verbose`</b> (bool, optional):  Whether to display progress. Defaults to True. 
 - <b>`replace_iou`</b> (function, optional):  IoU replacement function. Defaults to None. 



**Returns:**
 
------- 
 - <b>`tuple`</b>:  Tuple containing the recall, precision, and scores. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/metrics.py#L260"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `getAveragePrecision`

```python
getAveragePrecision(
    od_predictions: ODPredictions,
    verbose=True,
    iou_threshold=0.3
) → tuple
```

Get the average precision for object detection predictions. 



**Args:**
 
---- 
 - <b>`od_predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`verbose`</b> (bool, optional):  Whether to display progress. Defaults to True. 
 - <b>`iou_threshold`</b> (float, optional):  IoU threshold. Defaults to 0.3. 



**Returns:**
 
------- 
 - <b>`tuple`</b>:  Tuple containing the average precision, total recalls, total precisions, and objectness thresholds. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/metrics.py#L302"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_recall_precision`

```python
plot_recall_precision(
    total_recalls: list,
    total_precisions: list,
    threshes_objectness: ndarray
)
```

Plot the recall and precision given objectness threshold or IoU threshold. 



**Args:**
 
---- 
 - <b>`total_recalls`</b> (list):  List of total recalls. 
 - <b>`total_precisions`</b> (list):  List of total precisions. 
 - <b>`threshes_objectness`</b> (np.ndarray):  Array of objectness thresholds. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/metrics.py#L327"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `unroll_metrics`

```python
unroll_metrics(
    predictions: ODPredictions,
    conformalized_predictions: ODConformalizedPredictions,
    confidence_threshold: Optional[float, Tensor] = None,
    iou_threshold: float = 0.5,
    verbose: bool = True
) → dict
```

Compute and return various metrics for object detection predictions and conformalized predictions. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`conformalized_predictions`</b> (ODConformalizedPredictions):  Conformalized object detection predictions. 
 - <b>`confidence_threshold`</b> (float or torch.Tensor, optional):  Confidence threshold. Defaults to None. 
 - <b>`iou_threshold`</b> (float, optional):  IoU threshold. Defaults to 0.5. 
 - <b>`verbose`</b> (bool, optional):  Whether to display progress. Defaults to True. 



**Returns:**
 
------- 
 - <b>`dict`</b>:  Dictionary containing metrics such as AP, recalls, precisions, and thresholds. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/metrics.py#L398"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODEvaluator`
Evaluator for object detection predictions using specified loss functions. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/metrics.py#L401"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(confidence_loss, localization_loss, classification_loss)
```

Initialize the ODEvaluator. 



**Args:**
 
---- 
 - <b>`confidence_loss`</b> (callable):  Loss function for confidence. 
 - <b>`localization_loss`</b> (callable):  Loss function for localization. 
 - <b>`classification_loss`</b> (callable):  Loss function for classification. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/metrics.py#L420"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    predictions: ODPredictions,
    parameters: ODParameters,
    conformalized_predictions: ODConformalizedPredictions
)
```

Evaluate predictions using the provided loss functions and return results. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`parameters`</b> (ODParameters):  Parameters for object detection. 
 - <b>`conformalized_predictions`</b> (ODConformalizedPredictions):  Conformalized object detection predictions. 



**Returns:**
 
------- 
 - <b>`ODResults`</b>:  Results object containing computed losses and set sizes. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
