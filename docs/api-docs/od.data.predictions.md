<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.data.predictions`
Data structures for object detection predictions and results. 

This module defines the data structures used to store and manipulate object detection predictions, parameters, conformalized predictions, and evaluation results in the conformal prediction framework. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    dataset_name: 'str',
    split_name: 'str',
    image_paths: 'list[str]',
    image_shapes: 'list[Tensor]',
    true_boxes: 'list[Tensor]',
    pred_boxes: 'list[Tensor]',
    confidences: 'list[Tensor]',
    true_cls: 'list[Tensor]',
    pred_cls: 'list[Tensor]',
    names: 'list[str]',
    pred_boxes_uncertainty: 'list[Tensor]' = None,
    unique_id: 'int | None' = None
)
```

Initialize object detection predictions. 



**Args:**
 
 - <b>`dataset_name`</b> (str):  Name of the dataset. 
 - <b>`split_name`</b> (str):  Name of the dataset split. 
 - <b>`image_paths`</b> (list[str]):  List of image file paths. 
 - <b>`image_shapes`</b> (list[torch.Tensor]):  List of image shapes. 
 - <b>`true_boxes`</b> (list[torch.Tensor]):  List of ground truth bounding boxes. 
 - <b>`pred_boxes`</b> (list[torch.Tensor]):  List of predicted bounding boxes. 
 - <b>`confidences`</b> (list[torch.Tensor]):  List of confidence scores. 
 - <b>`true_cls`</b> (list[torch.Tensor]):  List of ground truth class labels. 
 - <b>`pred_cls`</b> (list[torch.Tensor]):  List of predicted class probabilities. 
 - <b>`names`</b> (list[str]):  List of class names. 
 - <b>`pred_boxes_uncertainty`</b> (list[torch.Tensor], optional):  List of prediction uncertainties. 
 - <b>`unique_id`</b> (int, optional):  Unique identifier for the predictions. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to`

```python
to(device: 'str')
```

Move the data to the specified device. 



**Args:**
 
 - <b>`device`</b> (str):  The device to move the data to. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODParameters`
Class representing parameters for object detection tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    global_alpha: 'float',
    confidence_threshold: 'float',
    predictions_id: 'int',
    alpha_confidence: 'float | None' = None,
    alpha_localization: 'float | None' = None,
    alpha_classification: 'float | None' = None,
    lambda_confidence_plus: 'float | None' = None,
    lambda_confidence_minus: 'float | None' = None,
    lambda_localization: 'float | None' = None,
    lambda_classification: 'float | None' = None,
    unique_id: 'int | None' = None
)
```

Initialize a new instance of the ODParameters class. 



**Args:**
 
 - <b>`global_alpha`</b> (float):  The global alpha (the sum of the non-None alphas). 
 - <b>`confidence_threshold`</b> (float):  The confidence threshold. 
 - <b>`predictions_id`</b> (int):  The unique ID of the predictions. 
 - <b>`alpha_confidence`</b> (float, optional):  The alpha for confidence. Defaults to None. 
 - <b>`alpha_localization`</b> (float, optional):  The alpha for localization. Defaults to None. 
 - <b>`alpha_classification`</b> (float, optional):  The alpha for classification. Defaults to None. 
 - <b>`lambda_confidence_plus`</b> (float, optional):  The lambda for confidence (conservative). Defaults to None. 
 - <b>`lambda_confidence_minus`</b> (float, optional):  The lambda for confidence (optimistic). Defaults to None. 
 - <b>`lambda_localization`</b> (float, optional):  The lambda for localization. Defaults to None. 
 - <b>`lambda_classification`</b> (float, optional):  The lambda for classification. Defaults to None. 
 - <b>`unique_id`</b> (int, optional):  The unique ID of the parameters. Defaults to None. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L196"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODConformalizedPredictions`
Class representing conformalized predictions for object detection tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L199"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    predictions: 'ODPredictions',
    parameters: 'ODParameters',
    conf_boxes: 'Tensor | None' = None,
    conf_cls: 'Tensor | None' = None
)
```

Initialize a new instance of the ODConformalizedPredictions class. 



**Args:**
 
 - <b>`predictions`</b> (ODPredictions):  The object detection predictions. 
 - <b>`parameters`</b> (ODParameters):  The conformalizers parameters. 
 - <b>`conf_boxes`</b> (torch.Tensor, optional):  The conformal boxes, after filtering. Defaults to None. 
 - <b>`conf_cls`</b> (torch.Tensor, optional):  The conformal prediction sets for class labels of each box. Defaults to None. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODResults`
Class representing results for object detection tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/predictions.py#L227"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    predictions: 'ODPredictions',
    parameters: 'Parameters',
    conformalized_predictions: 'ODConformalizedPredictions',
    confidence_set_sizes: 'Tensor | list[float] | None' = None,
    confidence_coverages: 'Tensor | list[float] | None' = None,
    localization_set_sizes: 'Tensor | list[float] | None' = None,
    localization_coverages: 'Tensor | list[float] | None' = None,
    classification_set_sizes: 'Tensor | list[float] | None' = None,
    classification_coverages: 'Tensor | list[float] | None' = None,
    global_coverage: 'Tensor | float | None' = None
)
```

Initialize a new instance of the ODResults class. 



**Args:**
 
 - <b>`predictions`</b> (ODPredictions):  The object detection predictions. 
 - <b>`parameters`</b> (Parameters):  The conformalizers parameters. 
 - <b>`conformalized_predictions`</b> (ODConformalizedPredictions):  The conformalized predictions. 
 - <b>`confidence_set_sizes`</b> (torch.Tensor | list[float], optional):  The confidence set sizes. Defaults to None. 
 - <b>`confidence_coverages`</b> (torch.Tensor | list[float], optional):  The confidence coverages. Defaults to None. 
 - <b>`localization_set_sizes`</b> (torch.Tensor | list[float], optional):  The localization set sizes. Defaults to None. 
 - <b>`localization_coverages`</b> (torch.Tensor | list[float], optional):  The localization coverages. Defaults to None. 
 - <b>`classification_set_sizes`</b> (torch.Tensor | list[float], optional):  The classification set sizes. Defaults to None. 
 - <b>`classification_coverages`</b> (torch.Tensor | list[float], optional):  The classification coverages. Defaults to None. 
 - <b>`global_coverage`</b> (torch.Tensor | float, optional):  The global coverage. Defaults to None. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
