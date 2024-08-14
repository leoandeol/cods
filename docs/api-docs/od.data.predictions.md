<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.data.predictions`






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODPredictions`
Class representing predictions for object detection tasks. 



**Args:**
 
 - <b>`id`</b> (int):  Unique ID of the predictions. 
 - <b>`dataset_name`</b> (str):  Name of the dataset. 
 - <b>`split_name`</b> (str):  Name of the data split. 
 - <b>`image_paths`</b>:  List of image paths. 
 - <b>`true_boxes`</b>:  List of true bounding boxes. 
 - <b>`pred_boxes`</b>:  List of predicted bounding boxes. 
 - <b>`confidences`</b>:  List of confidence scores for predicted boxes. 
 - <b>`true_cls`</b>:  List of true class labels. 
 - <b>`pred_cls`</b>:  List of predicted class labels. 



**Attributes:**
 
 - <b>`image_paths`</b>:  List of image paths. 
 - <b>`true_boxes`</b>:  List of true bounding boxes. 
 - <b>`pred_boxes`</b>:  List of predicted bounding boxes. 
 - <b>`confidence`</b>:  List of confidence scores for predicted boxes. 
 - <b>`true_cls`</b>:  List of true class labels. 
 - <b>`pred_cls`</b>:  List of predicted class labels. 
 - <b>`preds_cls`</b>:  ClassificationPredictions instance. 
 - <b>`n_classes`</b>:  Number of classes. 
 - <b>`matching`</b>:  Matching information. 
 - <b>`confidence_threshold`</b>:  Confidence threshold. 

Methods: 
 - <b>`__len__`</b>:  Returns the number of image paths. 
 - <b>`__str__`</b>:  Returns a string representation of the ODPredictions object. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    dataset_name: str,
    split_name: str,
    image_paths,
    true_boxes,
    pred_boxes,
    confidences,
    true_cls,
    pred_cls,
    unique_id: Optional[int] = None
)
```









---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODParameters`
Class representing parameters for object detection tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    global_alpha: float,
    alpha_confidence: Optional[float],
    alpha_localization: Optional[float],
    alpha_classification: Optional[float],
    lambda_confidence_plus: Optional[float],
    lambda_confidence_minus: Optional[float],
    lambda_localization: Optional[float],
    lambda_classification: Optional[float],
    confidence_threshold: float,
    predictions_id: int,
    unique_id: Optional[int] = None
)
```

Initializes a new instance of the ODParameters class. 



**Parameters:**
 
 - <b>`global_alpha`</b> (float):  The global alpha (the sum of the non-None alphas). 
 - <b>`alpha_confidence`</b> (float):  The alpha for confidence. 
 - <b>`alpha_localization`</b> (float):  The alpha for localization. 
 - <b>`alpha_classification`</b> (float):  The alpha for classification 
 - <b>`lambda_confidence_plus`</b> (float):  The lambda for confidence (conservative). 
 - <b>`lambda_confidence_minus`</b> (float):  The lambda for confidence (optimistic). 
 - <b>`lambda_localization`</b> (float):  The lambda for localization. 
 - <b>`lambda_classification`</b> (float):  The lambda for classification. 
 - <b>`confidence_threshold`</b> (float):  The confidence threshold. 
 - <b>`predictions_id`</b> (int):  The unique ID of the predictions. 
 - <b>`unique_id`</b> (int):  The unique ID of the parameters. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODConformalizedPredictions`
Class representing conformalized predictions for object detection tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    predictions: ODPredictions,
    parameters: ODParameters,
    conf_boxes: Optional[Tensor],
    conf_cls: Optional[Tensor]
)
```

Initializes a new instance of the ODResults class. 



**Parameters:**
 
 - <b>`predictions`</b> (ODPredictions):  The object detection predictions. 
 - <b>`parameters`</b> (ODParameters):  The conformalizers parameters. 
 - <b>`conf_boxes`</b> (torch.Tensor):  The conformal boxes. 
 - <b>`conf_cls`</b> (torch.Tensor):  The conformal prediction sets for class labels of each box. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODResults`
Class representing results for object detection tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    predictions: ODPredictions,
    parameters: Parameters,
    conformalized_predictions: ODConformalizedPredictions,
    confidence_set_sizes: Optional[Tensor, List[float]],
    confidence_coverages: Optional[Tensor, List[float]],
    localization_set_sizes: Optional[Tensor, List[float]],
    localization_coverages: Optional[Tensor, List[float]],
    classification_set_sizes: Optional[Tensor, List[float]],
    classification_coverages: Optional[Tensor, List[float]],
    global_coverage: Optional[Tensor, float] = None
)
```

Initializes a new instance of the ODResults class. 



**Parameters:**
 
 - <b>`predictions`</b> (ODPredictions):  The object detection predictions. 
 - <b>`parameters`</b> (ODParameters):  The conformalizers parameters. 
 - <b>`conformalized_predictions`</b> (ODConformalizedPredictions):  The conformalized predictions. 
 - <b>`confidence_set_sizes`</b> (torch.Tensor):  The confidence set sizes. 
 - <b>`confidence_coverages`</b> (torch.Tensor):  The confidence coverages. 
 - <b>`localization_set_sizes`</b> (torch.Tensor):  The localization set sizes. 
 - <b>`localization_coverages`</b> (torch.Tensor):  The localization coverages. 
 - <b>`classification_set_sizes`</b> (torch.Tensor):  The classification set sizes. 
 - <b>`classification_coverages`</b> (torch.Tensor):  The classification coverages. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
