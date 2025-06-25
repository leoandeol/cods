<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.cp`




**Global Variables**
---------------
- **FORMAT**


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LocalizationConformalizer`
A class for performing localization conformalization. Should be used within an ODConformalizer. 

Attributes 
----------  BACKENDS (list): Supported backends.  accepted_methods (dict): Mapping of accepted method names to score functions.  PREDICTION_SETS (list): Supported prediction sets.  LOSSES (dict): Mapping of loss names to loss classes.  OPTIMIZERS (dict): Mapping of optimizer names to optimizer classes.  GUARANTEE_LEVELS (list): Supported guarantee levels. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    loss: 'str | ODLoss',
    prediction_set: 'str',
    guarantee_level: 'str',
    matching_function: 'str',
    number_of_margins: 'int' = 1,
    optimizer: 'str | Optimizer | None' = None,
    backend: 'str' = 'auto',
    device: 'str' = 'cpu'
)
```

Initialize the LocalizationConformalizer. 



**Args:**
 
---- 
 - <b>`loss`</b> (Union[str, ODLoss]):  Loss function or its name. 
 - <b>`prediction_set`</b> (str):  Prediction set type. 
 - <b>`guarantee_level`</b> (str):  Guarantee level. 
 - <b>`matching_function`</b> (str):  Matching function name. 
 - <b>`number_of_margins`</b> (int, optional):  Number of margins. Defaults to 1. 
 - <b>`optimizer`</b> (Optional[Union[str, Optimizer]], optional):  Optimizer. Defaults to None. 
 - <b>`backend`</b> (str, optional):  Backend. Defaults to "auto". 
 - <b>`device`</b> (str, optional):  Device. Defaults to "cpu". 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    predictions: 'ODPredictions',
    alpha: 'float',
    steps: 'int' = 13,
    bounds: 'list[float]' = None,
    verbose: 'bool' = True,
    overload_confidence_threshold: 'float | None' = None
) → float
```

Calibrate the conformalizer for localization. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`alpha`</b> (float):  Significance level. 
 - <b>`steps`</b> (int, optional):  Number of optimization steps. Defaults to 13. 
 - <b>`bounds`</b> (List[float], optional):  Bounds for optimization. Defaults to [0, 1000]. 
 - <b>`verbose`</b> (bool, optional):  Verbosity. Defaults to True. 
 - <b>`overload_confidence_threshold`</b> (Optional[float], optional):  Overload confidence threshold. Defaults to None. 



**Returns:**
 
------- 
 - <b>`float`</b>:  Calibrated lambda value. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L282"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(
    predictions: 'ODPredictions',
    parameters: 'ODParameters | None' = None,
    verbose: 'bool' = True
) → list[Tensor]
```

Conformalize predictions using the calibrated lambda values for localization. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Predictions to conformalize. 
 - <b>`parameters`</b> (Optional[ODParameters], optional):  Parameters with lambda value. Defaults to None. 
 - <b>`verbose`</b> (bool, optional):  Verbosity. Defaults to True. 



**Returns:**
 
------- 
 - <b>`List[torch.Tensor]`</b>:  Conformalized bounding boxes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConfidenceConformalizer`
Conformalizer for confidence/objectness in object detection. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L349"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    guarantee_level: 'str',
    matching_function: 'str',
    loss: 'str' = 'box_count_threshold',
    other_losses: 'list | None' = None,
    optimizer: 'str' = 'binary_search',
    device='cpu'
)
```

Initialize the ConfidenceConformalizer. 



**Args:**
 
---- 
 - <b>`guarantee_level`</b> (str):  Guarantee level. 
 - <b>`matching_function`</b> (str):  Matching function name. 
 - <b>`loss`</b> (str, optional):  Loss name. Defaults to "box_count_threshold". 
 - <b>`other_losses`</b> (Optional[List], optional):  Other losses. Defaults to None. 
 - <b>`optimizer`</b> (str, optional):  Optimizer name. Defaults to "binary_search". 
 - <b>`device`</b> (str, optional):  Device. Defaults to "cpu". 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L402"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    predictions: 'ODPredictions',
    alpha: 'float' = 0.1,
    steps: 'int' = 13,
    bounds: 'list[float]' = None,
    verbose: 'bool' = True
) → tuple[float, float]
```

Calibrate the conformalizer for confidence/objectness. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`alpha`</b> (float, optional):  Significance level. Defaults to 0.1. 
 - <b>`steps`</b> (int, optional):  Number of optimization steps. Defaults to 13. 
 - <b>`bounds`</b> (List[float], optional):  Bounds for optimization. Defaults to [0, 1]. 
 - <b>`verbose`</b> (bool, optional):  Verbosity. Defaults to True. 



**Returns:**
 
------- 
 - <b>`Tuple[float, float]`</b>:  Calibrated lambda_minus and lambda_plus values. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L462"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(predictions: 'ODPredictions', verbose: 'bool' = True) → float
```

Conformalize the object detection predictions using calibrated lambda values. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`verbose`</b> (bool, optional):  Verbosity. Defaults to True. 



**Returns:**
 
------- 
 - <b>`float`</b>:  The new confidence threshold (1 - lambda_plus). 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L488"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODClassificationConformalizer`
Conformalizer for classification in object detection tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L501"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    matching_function: 'str',
    loss='binary',
    prediction_set='lac',
    backend='auto',
    guarantee_level='image',
    optimizer='binary_search',
    device='cpu'
)
```

Initialize the ODClassificationConformalizer. 



**Args:**
 
---- 
 - <b>`matching_function`</b> (str):  Matching function name. 
 - <b>`loss`</b> (str, optional):  Loss name. Defaults to "binary". 
 - <b>`prediction_set`</b> (str, optional):  Prediction set. Defaults to "lac". 
 - <b>`backend`</b> (str, optional):  Backend. Defaults to "auto". 
 - <b>`guarantee_level`</b> (str, optional):  Guarantee level. Defaults to "image". 
 - <b>`optimizer`</b> (str, optional):  Optimizer name. Defaults to "binary_search". 
 - <b>`device`</b> (str, optional):  Device. Defaults to "cpu". 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L574"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    predictions: 'ODPredictions',
    alpha: 'float',
    bounds: 'list[float] | None' = None,
    steps: 'int' = 40,
    verbose: 'bool' = True,
    overload_confidence_threshold: 'float | None' = None
) → Tensor
```

Calibrate the conformalizer for classification. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`alpha`</b> (float):  Significance level. 
 - <b>`bounds`</b> (List[float], optional):  Bounds for optimization. Defaults to [0, 1]. 
 - <b>`steps`</b> (int, optional):  Number of optimization steps. Defaults to 40. 
 - <b>`verbose`</b> (bool, optional):  Verbosity. Defaults to True. 
 - <b>`overload_confidence_threshold`</b> (Optional[float], optional):  Overload confidence threshold. Defaults to None. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Calibrated lambda value for classification. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L670"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(predictions: 'ODPredictions', verbose: 'bool' = True) → list
```

Conformalize the predictions for classification. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`verbose`</b> (bool, optional):  Verbosity. Defaults to True. 



**Returns:**
 
------- 
 - <b>`List`</b>:  Conformalized class predictions. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L707"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODConformalizer`
Class representing conformalizers for object detection tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L715"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    backend: 'str' = 'auto',
    guarantee_level: 'str' = 'image',
    matching_function: 'str' = 'hausdorff',
    confidence_threshold: 'float | None' = None,
    multiple_testing_correction: 'str | None' = None,
    confidence_method: 'ConfidenceConformalizer | str | None' = None,
    localization_method: 'LocalizationConformalizer | str | None' = None,
    localization_prediction_set: 'str' = 'additive',
    classification_method: 'ClassificationConformalizer | str | None' = None,
    classification_prediction_set: 'str' = 'lac',
    optimizer='binary_search',
    device='cpu'
)
```

Initialize the ODConformalizer object. 



**Args:**
 
---- 
 - <b>`backend`</b> (str, optional):  Backend. Defaults to "auto". 
 - <b>`guarantee_level`</b> (str, optional):  Guarantee level. Defaults to "image". 
 - <b>`matching_function`</b> (str, optional):  Matching function. Defaults to "hausdorff". 
 - <b>`confidence_threshold`</b> (Optional[float], optional):  Confidence threshold. Defaults to None. 
 - <b>`multiple_testing_correction`</b> (Optional[str], optional):  Multiple testing correction. Defaults to None. 
 - <b>`confidence_method`</b> (Union[ConfidenceConformalizer, str, None], optional):  Confidence conformalizer or method. Defaults to None. 
 - <b>`localization_method`</b> (Union[LocalizationConformalizer, str, None], optional):  Localization conformalizer or method. Defaults to None. 
 - <b>`localization_prediction_set`</b> (str, optional):  Localization prediction set. Defaults to "additive". 
 - <b>`classification_method`</b> (Union[ClassificationConformalizer, str, None], optional):  Classification conformalizer or method. Defaults to None. 
 - <b>`classification_prediction_set`</b> (str, optional):  Classification prediction set. Defaults to "lac". 
 - <b>`optimizer`</b> (str, optional):  Optimizer. Defaults to "binary_search". 
 - <b>`device`</b> (str, optional):  Device. Defaults to "cpu". 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L886"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    predictions: 'ODPredictions',
    global_alpha: 'float | None' = None,
    alpha_confidence: 'float | None' = None,
    alpha_localization: 'float | None' = None,
    alpha_classification: 'float | None' = None,
    verbose: 'bool' = True
) → ODParameters
```

Calibrates the conformalizers and returns the calibration results. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Predictions to calibrate. 
 - <b>`global_alpha`</b> (Optional[float], optional):  Global alpha for calibration. Defaults to None. 
 - <b>`alpha_confidence`</b> (Optional[float], optional):  Alpha for confidence. Defaults to None. 
 - <b>`alpha_localization`</b> (Optional[float], optional):  Alpha for localization. Defaults to None. 
 - <b>`alpha_classification`</b> (Optional[float], optional):  Alpha for classification. Defaults to None. 
 - <b>`verbose`</b> (bool, optional):  Verbosity. Defaults to True. 



**Returns:**
 
------- 
 - <b>`ODParameters`</b>:  Calibration results. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1098"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(
    predictions: 'ODPredictions',
    parameters: 'ODParameters | None' = None,
    verbose: 'bool' = True
) → ODConformalizedPredictions
```

Conformalize the given predictions using the provided parameters. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Predictions to conformalize. 
 - <b>`parameters`</b> (Optional[ODParameters], optional):  Parameters for conformalization. Defaults to None. 
 - <b>`verbose`</b> (bool, optional):  Verbosity. Defaults to True. 



**Returns:**
 
------- 
 - <b>`ODConformalizedPredictions`</b>:  Conformalized predictions. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    predictions: 'ODPredictions',
    parameters: 'ODParameters',
    conformalized_predictions: 'ODConformalizedPredictions',
    include_confidence_in_global: 'bool',
    verbose: 'bool' = True
) → ODResults
```

Evaluate the conformalized predictions and return results. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Predictions to evaluate. 
 - <b>`parameters`</b> (ODParameters):  Parameters used for conformalization. 
 - <b>`conformalized_predictions`</b> (ODConformalizedPredictions):  Conformalized predictions. 
 - <b>`include_confidence_in_global`</b> (bool):  Whether to include confidence in global coverage. 
 - <b>`verbose`</b> (bool, optional):  Verbosity. Defaults to True. 



**Returns:**
 
------- 
 - <b>`ODResults`</b>:  Evaluation results. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1277"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AsymptoticLocalizationObjectnessConformalizer`
A class that performs risk conformalization for object detection predictions with asymptotic localization and objectness. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    prediction_set: 'str' = 'additive',
    localization_loss: 'str' = 'boxwise',
    optimizer: 'str' = 'gpr'
)
```

Initialize the AsymptoticLocalizationObjectnessConformalizer. 



**Args:**
 
---- 
 - <b>`prediction_set`</b> (str, optional):  Prediction set type. Defaults to "additive". 
 - <b>`localization_loss`</b> (str, optional):  Localization loss type. Defaults to "boxwise". 
 - <b>`optimizer`</b> (str, optional):  Optimizer type. Defaults to "gpr". 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1384"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    predictions: 'ODPredictions',
    alpha: 'float' = 0.1,
    steps: 'int' = 13,
    bounds: 'list' = None,
    verbose: 'bool' = True
)
```

Calibrate the conformalizer using the given predictions. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`alpha`</b> (float, optional):  Significance level. Defaults to 0.1. 
 - <b>`steps`</b> (int, optional):  Number of optimization steps. Defaults to 13. 
 - <b>`bounds`</b> (list, optional):  Bounds for optimization. Defaults to [(0, 500), (0.0, 1.0)]. 
 - <b>`verbose`</b> (bool, optional):  Verbosity. Defaults to True. 



**Returns:**
 
------- 
 - <b>`tuple`</b>:  Calibrated lambda values. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1426"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(predictions: 'ODPredictions', verbose: 'bool' = True)
```

Conformalize predictions using the calibrated lambda values. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`verbose`</b> (bool, optional):  Verbosity. Defaults to True. 



**Returns:**
 
------- 
 - <b>`list`</b>:  Conformalized bounding boxes. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1453"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    predictions: 'ODPredictions',
    conf_boxes: 'list',
    verbose: 'bool' = True
)
```

Evaluate the conformalized predictions. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`conf_boxes`</b> (list):  Conformalized bounding boxes. 
 - <b>`verbose`</b> (bool, optional):  Verbosity. Defaults to True. 



**Returns:**
 
------- 
 - <b>`tuple`</b>:  Evaluation results. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
