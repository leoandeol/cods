<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.data.predictions`






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODPredictions`
Class representing predictions for object detection tasks. 



**Args:**
 
---- 
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
 
---------- 
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
------- 
 - <b>`__len__`</b>:  Returns the number of image paths. 
 - <b>`__str__`</b>:  Returns a string representation of the ODPredictions object. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    dataset_name: str,
    split_name: str,
    image_paths: List[str],
    image_shapes: List[Tensor],
    true_boxes: List[Tensor],
    pred_boxes: List[Tensor],
    confidences: List[Tensor],
    true_cls: List[Tensor],
    pred_cls: List[Tensor],
    names: List[str],
    pred_boxes_uncertainty: List[Tensor] = None,
    unique_id: Optional[int] = None
)
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to`

```python
to(device: str)
```

Move the data to the specified device. 

Parameters 
----------  device (str): The device to move the data to. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODParameters`
Class representing parameters for object detection tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    global_alpha: float,
    confidence_threshold: float,
    predictions_id: int,
    alpha_confidence: Optional[float] = None,
    alpha_localization: Optional[float] = None,
    alpha_classification: Optional[float] = None,
    lambda_confidence_plus: Optional[float] = None,
    lambda_confidence_minus: Optional[float] = None,
    lambda_localization: Optional[float] = None,
    lambda_classification: Optional[float] = None,
    unique_id: Optional[int] = None
)
```

Initializes a new instance of the ODParameters class. 

Parameters 
----------  global_alpha (float): The global alpha (the sum of the non-None alphas).  alpha_confidence (float): The alpha for confidence.  alpha_localization (float): The alpha for localization.  alpha_classification (float): The alpha for classification  lambda_confidence_plus (float): The lambda for confidence (conservative).  lambda_confidence_minus (float): The lambda for confidence (optimistic).  lambda_localization (float): The lambda for localization.  lambda_classification (float): The lambda for classification.  confidence_threshold (float): The confidence threshold.  predictions_id (int): The unique ID of the predictions.  unique_id (int): The unique ID of the parameters. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODConformalizedPredictions`
Class representing conformalized predictions for object detection tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    predictions: ODPredictions,
    parameters: ODParameters,
    conf_boxes: Optional[Tensor] = None,
    conf_cls: Optional[Tensor] = None
)
```

Initializes a new instance of the ODResults class. 

Parameters 
----------  predictions (ODPredictions): The object detection predictions.  parameters (ODParameters): The conformalizers parameters.  conf_boxes (torch.Tensor): The conformal boxes, after filtering.  conf_cls (torch.Tensor): The conformal prediction sets for class labels of each box. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODResults`
Class representing results for object detection tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    predictions: ODPredictions,
    parameters: Parameters,
    conformalized_predictions: ODConformalizedPredictions,
    confidence_set_sizes: Optional[Tensor, List[float]] = None,
    confidence_coverages: Optional[Tensor, List[float]] = None,
    localization_set_sizes: Optional[Tensor, List[float]] = None,
    localization_coverages: Optional[Tensor, List[float]] = None,
    classification_set_sizes: Optional[Tensor, List[float]] = None,
    classification_coverages: Optional[Tensor, List[float]] = None,
    global_coverage: Optional[Tensor, float] = None
)
```

Initializes a new instance of the ODResults class. 

Parameters 
----------  predictions (ODPredictions): The object detection predictions.  parameters (ODParameters): The conformalizers parameters.  conformalized_predictions (ODConformalizedPredictions): The conformalized predictions.  confidence_set_sizes (torch.Tensor): The confidence set sizes.  confidence_coverages (torch.Tensor): The confidence coverages.  localization_set_sizes (torch.Tensor): The localization set sizes.  localization_coverages (torch.Tensor): The localization coverages.  classification_set_sizes (torch.Tensor): The classification set sizes.  classification_coverages (torch.Tensor): The classification coverages.  global_coverage (torch.Tensor | float): The global coverage. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
