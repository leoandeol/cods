<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.cp`






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LocalizationConformalizer`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(method: str = 'min-hausdorff-additive', margins: int = 1, **kwargs)
```

Conformalizer for object localization tasks. 



**Args:**
 
 - <b>`method`</b> (str):  The method to compute non-conformity scores. Must be one of ["min-hausdorff-additive", "min-hausdorff-multiplicative", "union-hausdorff-additive", "union-hausdorff-multiplicative"]. 
 - <b>`margins`</b> (int):  The number of margins to compute. Must be one of [1, 2, 4]. 
 - <b>`**kwargs`</b>:  Additional keyword arguments. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    preds: ODPredictions,
    alpha: float = 0.1,
    confidence_threshold: Optional[float] = None,
    verbose: bool = True
) → list
```

Calibrates the conformalizer using the given predictions. 



**Args:**
 
 - <b>`preds`</b> (ODPredictions):  The object detection predictions. 
 - <b>`alpha`</b> (float):  The significance level for the calibration. 
 - <b>`confidence_threshold`</b> (float, optional):  The confidence threshold for the predictions. If not provided, it must be set in the predictions or in the conformalizer. 
 - <b>`verbose`</b> (bool):  Whether to display progress information. 



**Returns:**
 
 - <b>`list`</b>:  The computed quantiles for each margin. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L198"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds: ODPredictions) → list
```

Conformalizes the object detection predictions using the calibrated quantiles. 



**Args:**
 
 - <b>`preds`</b> (ODPredictions):  The object detection predictions. 



**Returns:**
 
 - <b>`list`</b>:  The conformalized bounding boxes. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L217"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    preds: ODPredictions,
    conf_boxes: list,
    verbose: bool = True
) → Tuple[Tensor, Tensor]
```

Evaluates the conformalized predictions. 



**Args:**
 
 - <b>`preds`</b> (ODPredictions):  The object detection predictions. 
 - <b>`conf_boxes`</b> (list):  The conformalized bounding boxes. 
 - <b>`verbose`</b> (bool):  Whether to display evaluation results. 



**Returns:**
 
 - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  The computed coverage and set sizes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LocalizationRiskConformalizer`
A class that performs risk conformalization for localization tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L281"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    prediction_set: str = 'additive',
    loss: str = None,
    optimizer: str = 'binary_search'
)
```

Initialize the LocalizationRiskConformalizer. 



**Parameters:**
 
- prediction_set (str): The type of prediction set to use. Must be one of ["additive", "multiplicative", "adaptative"]. 
- loss (str): The type of loss to use. Must be one of ["pixelwise", "boxwise"]. 
- optimizer (str): The type of optimizer to use. Must be one of ["binary_search", "gaussianprocess", "gpr", "kriging"]. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L381"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    preds: ODPredictions,
    alpha: float = 0.1,
    steps: int = 13,
    bounds: List[float] = [0, 1000],
    verbose: bool = True,
    confidence_threshold: float = None
) → float
```

Calibrate the conformalizer. 



**Parameters:**
 
- preds (ODPredictions): The object detection predictions. 
- alpha (float): The significance level. 
- steps (int): The number of steps for optimization. 
- bounds (List[float]): The bounds for optimization. 
- verbose (bool): Whether to print the optimization progress. 
- confidence_threshold (float): The threshold for objectness confidence. 



**Returns:**
 
- lbd (float): The calibrated lambda value. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L430"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds: ODPredictions) → List[List[float]]
```

Conformalize the object detection predictions. 



**Parameters:**
 
- preds (ODPredictions): The object detection predictions. 



**Returns:**
 
- conf_boxes (List[List[float]]): The conformalized bounding boxes. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L449"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    preds: ODPredictions,
    conf_boxes: List[List[float]],
    verbose: bool = True
) → Tuple[Tensor, Tensor]
```

Evaluate the conformalized predictions. 



**Parameters:**
 
- preds (ODPredictions): The object detection predictions. 
- conf_boxes (List[List[float]]): The conformalized bounding boxes. 
- verbose (bool): Whether to print the evaluation results. 



**Returns:**
 
- safety (torch.Tensor): The safety scores. 
- set_sizes (torch.Tensor): The set sizes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L502"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConfidenceConformalizer`
A class that performs risk conformalization for localization tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L509"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    prediction_set: str = 'additive',
    loss: str = 'nb_boxes',
    optimizer: str = 'binary_search'
)
```

Initialize the LocalizationRiskConformalizer. 



**Parameters:**
 
- prediction_set (str): The type of prediction set to use. Must be one of ["additive", "multiplicative", "adaptative"]. 
- loss (str): The type of loss to use. Must be one of ["pixelwise", "boxwise"]. 
- optimizer (str): The type of optimizer to use. Must be one of ["binary_search", "gaussianprocess", "gpr", "kriging"]. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L600"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    preds: ODPredictions,
    alpha: float = 0.1,
    steps: int = 13,
    bounds: List[float] = [0, 1000],
    verbose: bool = True,
    confidence_threshold: float = None
) → float
```

Calibrate the conformalizer. 



**Parameters:**
 
- preds (ODPredictions): The object detection predictions. 
- alpha (float): The significance level. 
- steps (int): The number of steps for optimization. 
- bounds (List[float]): The bounds for optimization. 
- verbose (bool): Whether to print the optimization progress. 
- confidence_threshold (float): The threshold for objectness confidence. 



**Returns:**
 
- lbd (float): The calibrated lambda value. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L649"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds: ODPredictions) → float
```

Conformalize the object detection predictions. 



**Parameters:**
 
- preds (ODPredictions): The object detection predictions. 



**Returns:**
 
- conf_boxes (List[List[float]]): The conformalized bounding boxes. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L664"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    preds: ODPredictions,
    conf_boxes: List[List[float]],
    verbose: bool = True
) → Tuple[Tensor, Tensor]
```

Evaluate the conformalized predictions. 



**Parameters:**
 
- preds (ODPredictions): The object detection predictions. 
- conf_boxes (List[List[float]]): The conformalized bounding boxes. 
- verbose (bool): Whether to print the evaluation results. 



**Returns:**
 
- safety (torch.Tensor): The safety scores. 
- set_sizes (torch.Tensor): The set sizes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L717"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODClassificationConformalizer`








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L722"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODConformalizer`
Class representing conformalizers for object detection tasks. 



**Attributes:**
 
 - <b>`MULTIPLE_TESTING_CORRECTIONS`</b> (List[str]):  List of supported multiple testing correction methods. 
 - <b>`BACKENDS`</b> (List[str]):  List of supported backends. 
 - <b>`GUARANTEE_LEVELS`</b> (List[str]):  List of supported guarantee levels. 



**Args:**
 
 - <b>`backend`</b> (str):  The backend used for the conformalization. Only 'auto' is supported currently. 
 - <b>`guarantee_level`</b> (str):  The guarantee level for the conformalization. Must be one of ["image", "object"]. 
 - <b>`confidence_threshold`</b> (Optional[float]):  The confidence threshold used for objectness conformalization. Mutually exclusive with 'confidence_method', if set, then confidence_method must be None. 
 - <b>`multiple_testing_correction`</b> (Optional[str]):  The method used for multiple testing correction. Must be one of ["bonferroni"] or None. None implies no correction is applied, and that a List of Alphas is expected for calibration instead. 
 - <b>`confidence_method`</b> (Union[ConfidenceConformalizer, str, None]):  The method used for confidence conformalization. Mutually exclusive with 'confidence_threshold', if set, then confidence_threshold must be None. Either pass a ConfidenceConformalizer instance, a string representing the method (loss) name, or None. 
 - <b>`localization_method`</b> (Union[LocalizationConformalizer, str, None]):  The method used for localization conformalization. Either pass a LocalizationConformalizer instance, a string representing the method (loss) name, or None. 
 - <b>`classification_method`</b> (Union[ClassificationConformalizer, str, None]):  The method used for classification conformalization. Either pass a ClassificationConformalizer instance, a string representing the method (loss) name, or None. 
 - <b>`**kwargs`</b>:  Additional keyword arguments. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the provided backend is not supported. 
 - <b>`ValueError`</b>:  If the provided guarantee level is not supported. 
 - <b>`ValueError`</b>:  If both confidence_threshold and confidence_method are provided. 
 - <b>`ValueError`</b>:  If neither confidence_threshold nor confidence_method are provided. 
 - <b>`ValueError`</b>:  If the provided multiple_testing_correction is not supported. 

Methods: calibrate(predictions, global_alpha, alpha_confidence, alpha_localization, alpha_classification, verbose=True)  Calibrates the conformalizers and returns the calibration results. 



**Args:**
 
         - <b>`predictions`</b> (ODPredictions):  The predictions to be calibrated. 
         - <b>`global_alpha`</b> (Optional[float]):  The global alpha value for calibration. If multiple_testing_correction is None, individual alpha values will be used for each conformalizer. 
         - <b>`alpha_confidence`</b> (Optional[float]):  The alpha value for the confidence conformalizer. 
         - <b>`alpha_localization`</b> (Optional[float]):  The alpha value for the localization conformalizer. 
         - <b>`alpha_classification`</b> (Optional[float]):  The alpha value for the classification conformalizer. 
         - <b>`verbose`</b> (bool, optional):  Whether to print calibration information. Defaults to True. 



**Returns:**
 
         - <b>`dict[str, Any]`</b>:  A dictionary containing the calibration results, including target alpha values and estimated lambda values for each conformalizer. 



**Raises:**
 
         - <b>`ValueError`</b>:  If the multiple_testing_correction is not provided or is not valid. 
         - <b>`ValueError`</b>:  If the global_alpha is not provided when using the Bonferroni multiple_testing_correction. 
         - <b>`ValueError`</b>:  If explicit alpha values are provided when using the Bonferroni multiple_testing_correction. 



**Note:**

> - The multiple_testing_correction attribute of the class must be set before calling this method. - The conformalizers must be initialized before calling this method. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L777"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    backend: str = 'auto',
    guarantee_level: str = 'image',
    confidence_threshold: Optional[float] = None,
    multiple_testing_correction: Optional[str] = None,
    confidence_method: Optional[ConfidenceConformalizer, str] = None,
    localization_method: Optional[LocalizationConformalizer, str] = None,
    classification_method: Optional[ClassificationConformalizer, str] = None,
    **kwargs
)
```

Initialize the ODClassificationConformalizer object. 



**Parameters:**
 
- backend (str): The backend used for the conformalization. Only 'auto' is supported currently. 
- guarantee_level (str): The guarantee level for the conformalization. Must be one of ["image", "object"]. 
- confidence_threshold (Optional[float]): The confidence threshold used for objectness conformalization.  Mutually exclusive with 'confidence_method', if set, then confidence_method must be None. 
- multiple_testing_correction (str): The method used for multiple testing correction. Must be one of ["bonferroni"] or None. None implies no correction is applied, and that a List of Alphas is expected for calibration instead. 
- confidence_method (Union[ConfidenceConformalizer, str, None]): The method used for confidence conformalization. Mutually exclusive with 'confidence_threshold', if set, then confidence_threshold must be None. Either pass a ConfidenceConformalizer instance, a string representing the method (loss) name, or None. 
- localization_method (Union[LocalizationConformalizer, str, None]): The method used for localization conformalization. Either pass a LocalizationConformalizer instance, a string representing the method (loss) name, or None. 
- classification_method (Union[ClassificationConformalizer, str, None]): The method used for classification conformalization. Either pass a ClassificationConformalizer instance, a string representing the method (loss) name, or None. 
- kwargs: Additional keyword arguments. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L883"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    predictions: ODPredictions,
    global_alpha: Optional[float],
    alpha_confidence: Optional[float],
    alpha_localization: Optional[float],
    alpha_classification: Optional[float],
    verbose: bool = True
) → ODParameters
```

Calibrates the conformalizers and returns the calibration results. 



**Args:**
 
 - <b>`predictions`</b> (ODPredictions):  The predictions to be calibrated. 
 - <b>`global_alpha`</b> (Optional[float]):  The global alpha value for calibration. If multiple_testing_correction is None, individual alpha values will be used for each conformalizer. 
 - <b>`alpha_confidence`</b> (Optional[float]):  The alpha value for the confidence conformalizer. 
 - <b>`alpha_localization`</b> (Optional[float]):  The alpha value for the localization conformalizer. 
 - <b>`alpha_classification`</b> (Optional[float]):  The alpha value for the classification conformalizer. 
 - <b>`verbose`</b> (bool, optional):  Whether to print calibration information. Defaults to True. 



**Returns:**
 
 - <b>`dict[str, Any]`</b>:  A dictionary containing the calibration results, including target alpha values and estimated lambda values for each conformalizer. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the multiple_testing_correction is not provided or is not valid. 
 - <b>`ValueError`</b>:  If the global_alpha is not provided when using the Bonferroni multiple_testing_correction. 
 - <b>`ValueError`</b>:  If explicit alpha values are provided when using the Bonferroni multiple_testing_correction. 



**Note:**

> - The multiple_testing_correction attribute of the class must be set before calling this method. - The conformalizers must be initialized before calling this method. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1067"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(
    predictions: ODPredictions,
    parameters=typing.Optional[cods.od.data.predictions.ODParameters],
    verbose: bool = True
) → ODConformalizedPredictions
```

Conformalize the given predictions. 



**Args:**
 
 - <b>`predictions`</b> (ODPredictions):  The predictions to be conformalized. 
 - <b>`parameters`</b> (Optional[ODParameters]):  The parameters to be used for conformalization. If None, the last parameters will be used. 
 - <b>`verbose`</b> (bool):  Whether to print conformalization information. 



**Returns:**
 
 - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  A tuple containing the conformalized predictions. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    predictions: ODPredictions,
    parameters: ODParameters,
    conformalized_predictions: ODConformalizedPredictions,
    include_confidence_in_global: bool,
    verbose: bool = True
) → ODResults
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SeqGlobalODRiskConformalizer`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1243"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    localization_method: str,
    objectness_method: str,
    classification_method: str,
    confidence_threshold: Optional[float] = None,
    fix_cls=False,
    **kwargs
)
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    preds: ODPredictions,
    alpha: float = 0.1,
    verbose: bool = True
) → Tuple[Sequence[float], float, float]
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1067"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(
    predictions: ODPredictions,
    parameters=typing.Optional[cods.od.data.predictions.ODParameters],
    verbose: bool = True
) → ODConformalizedPredictions
```

Conformalize the given predictions. 



**Args:**
 
 - <b>`predictions`</b> (ODPredictions):  The predictions to be conformalized. 
 - <b>`parameters`</b> (Optional[ODParameters]):  The parameters to be used for conformalization. If None, the last parameters will be used. 
 - <b>`verbose`</b> (bool):  Whether to print conformalization information. 



**Returns:**
 
 - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  A tuple containing the conformalized predictions. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1365"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    preds: ODPredictions,
    conf_boxes: list,
    conf_cls: list,
    verbose: bool = True
)
```

Evaluate the conformalizers. 



**Parameters:**
 
- preds: The ODPredictions object containing the predictions. 
- conf_boxes: The conformalized bounding boxes. 
- conf_cls: The conformalized classification scores. 
- verbose: Whether to print the evaluation results. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1438"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AsymptoticLocalizationObjectnessRiskConformalizer`
A class that performs risk conformalization for object detection predictions with asymptotic localization and objectness. 



**Args:**
 
 - <b>`prediction_set`</b> (str):  The type of prediction set to use. Must be one of "additive", "multiplicative", or "adaptative". 
 - <b>`localization_loss`</b> (str):  The type of localization loss to use. Must be one of "pixelwise" or "boxwise". 
 - <b>`optimizer`</b> (str):  The type of optimizer to use. Must be one of "gaussianprocess", "gpr", "kriging", "mc", or "montecarlo". 



**Attributes:**
 
 - <b>`ACCEPTED_LOSSES`</b> (dict):  A dictionary mapping accepted localization losses to their corresponding classes. 
 - <b>`loss_name`</b> (str):  The name of the localization loss. 
 - <b>`loss`</b> (Loss):  An instance of the localization loss class. 
 - <b>`prediction_set`</b> (str):  The type of prediction set. 
 - <b>`lbd`</b> (tuple):  The calibrated lambda values. 

Methods: 
 - <b>`_get_risk_function`</b>:  Returns the risk function for optimization. 
 - <b>`_correct_risk`</b>:  Corrects the risk using the number of predictions and the upper bound of the loss. 
 - <b>`calibrate`</b>:  Calibrates the conformalizer using the given predictions. 
 - <b>`conformalize`</b>:  Conformalizes the predictions using the calibrated lambda values. 
 - <b>`evaluate`</b>:  Evaluates the conformalized predictions. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1465"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    prediction_set: str = 'additive',
    localization_loss: str = 'boxwise',
    optimizer: str = 'gpr'
)
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1552"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    preds: ODPredictions,
    alpha: float = 0.1,
    steps: int = 13,
    bounds: list = [(0, 500), (0.0, 1.0)],
    verbose: bool = True
)
```

Calibrates the conformalizer using the given predictions. 



**Args:**
 
 - <b>`preds`</b> (ODPredictions):  The object detection predictions. 
 - <b>`alpha`</b> (float):  The significance level. 
 - <b>`steps`</b> (int):  The number of optimization steps. 
 - <b>`bounds`</b> (list):  The bounds for the optimization variables. 
 - <b>`verbose`</b> (bool):  Whether to print verbose output. 



**Returns:**
 
 - <b>`tuple`</b>:  The calibrated lambda values. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the conformalizer has already been calibrated. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1594"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds: ODPredictions)
```

Conformalizes the predictions using the calibrated lambda values. 



**Args:**
 
 - <b>`preds`</b> (ODPredictions):  The object detection predictions. 



**Returns:**
 
 - <b>`list`</b>:  The conformalized bounding boxes. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the conformalizer has not been calibrated. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1617"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(preds: ODPredictions, conf_boxes: list, verbose: bool = True)
```

Evaluates the conformalized predictions. 



**Args:**
 
 - <b>`preds`</b> (ODPredictions):  The object detection predictions. 
 - <b>`conf_boxes`</b> (list):  The conformalized bounding boxes. 
 - <b>`verbose`</b> (bool):  Whether to print verbose output. 



**Returns:**
 
 - <b>`tuple`</b>:  The evaluation results. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the conformalizer has not been calibrated or the predictions have not been conformalized. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
