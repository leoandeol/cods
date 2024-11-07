<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.cp`




**Global Variables**
---------------
- **FORMAT**


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LocalizationConformalizer`
A class for performing localization conformalization. Should be used within an ODConformalizer. 

Attributes 
---------- 
- BACKENDS (list): A list of supported backends. 
- accepted_methods (dict): A dictionary mapping accepted method names to their corresponding score functions. 
- PREDICTION_SETS (list): A list of supported prediction sets. 
- LOSSES (dict): A dictionary mapping loss names to their corresponding loss classes. 
- OPTIMIZERS (dict): A dictionary mapping optimizer names to their corresponding optimizer classes. 

Methods 
------- 
- __init__: Initialize the LocalizationConformalizer class. 
- _get_risk_function: Get the risk function for risk conformalization. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    loss: Union[str, ODLoss],
    prediction_set: str,
    guarantee_level: str,
    number_of_margins: int = 1,
    optimizer: Optional[str, Optimizer] = None,
    backend: str = 'auto',
    device='cpu',
    **kwargs
)
```

Initialize the CP class. 

Parameters 
---------- 
- loss (Union[str, ODLoss]): The loss function to be used. It can be either a string representing a predefined loss function or an instance of the ODLoss class. 
- prediction_set (str): The prediction set to be used. Must be one of ["additive", "multiplicative", "adaptive"]. 
- guarantee_level (str): The guarantee level to be used. Must be one of ["image", "object"]. 
- number_of_margins (int, optional): The number of margins to compute. Default is 1. 
- optimizer (Optional[Union[str, Optimizer]], optional): The optimizer to be used. It can be either a string representing a predefined optimizer or an instance of the Optimizer class. Default is None. 
- backend (str, optional): The backend to be used. Default is "auto". 
- **kwargs: Additional keyword arguments. 

Raises 
------ 
- ValueError: If the loss is not accepted, it must be one of the predefined losses or an instance of ODLoss. 
- ValueError: If the prediction set is not accepted, it must be one of the predefined prediction sets. 
- ValueError: If the number of margins is not 1, 2, or 4. 
- NotImplementedError: If the number of margins is greater than 1 (only 1 margin is supported for now). 
- ValueError: If the backend is not accepted, it must be one of the predefined backends. 
- ValueError: If the optimizer is not accepted, it must be one of the predefined optimizers or an instance of Optimizer. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L409"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    predictions: ODPredictions,
    alpha: float,
    steps: int = 13,
    bounds: List[float] = [0, 1000],
    verbose: bool = True,
    overload_confidence_threshold: Optional[float] = None
) → float
```

Calibrate the conformalizer. 

Parameters 
---------- 
- predictions (ODPredictions): The object detection predictions. 
- alpha (float): The significance level. 
- steps (int): The number of steps for optimization. 
- bounds (List[float]): The bounds for optimization. 
- verbose (bool): Whether to print the optimization progress. 
- confidence_threshold (float): The threshold for objectness confidence. 

Returns 
------- 
- lbd (float): The calibrated lambda value. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L475"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(
    predictions: ODPredictions,
    parameters: Optional[ODParameters] = None,
    verbose: bool = True
) → List[Tensor]
```

Conformalizes the predictions using the specified lambda values for localization. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  The predictions to be conformalized. 
 - <b>`parameters`</b> (Optional[ODParameters], optional):  The optional parameters containing the lambda value for localization. Defaults to None. 
 - <b>`verbose`</b> (bool, optional):  Whether to display verbose information. Defaults to True. 



**Returns:**
 
------- 
 - <b>`List[torch.Tensor]`</b>:  The conformalized bounding boxes. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If the conformalizer is not calibrated before conformalizing. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L536"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    predictions: ODPredictions,
    parameters: ODParameters,
    conformalized_predictions: ODConformalizedPredictions,
    verbose: bool = True
) → Tuple[Tensor, Tensor]
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L586"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConfidenceConformalizer`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L594"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    guarantee_level: str,
    matching_function: str,
    loss: str = 'nb_boxes',
    other_losses: Optional[List] = None,
    optimizer: str = 'binary_search',
    device='cpu'
)
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L761"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    predictions: ODPredictions,
    alpha: float = 0.1,
    steps: int = 13,
    bounds: List[float] = [0, 1],
    verbose: bool = True
) → Tuple[float, float]
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L805"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(predictions: ODPredictions) → float
```

Conformalize the object detection predictions. 

Parameters 
---------- 
- predictions (ODPredictions): The object detection predictions. 

Returns 
------- 
- conf_boxes (List[List[float]]): The conformalized bounding boxes. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L824"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    predictions: ODPredictions,
    parameters: ODParameters,
    conformalized_predictions: ODConformalizedPredictions,
    verbose: bool = True
) → Tuple[Tensor, Tensor]
```

Evaluate the conformalized predictions. 

Parameters 
---------- 
- predictions (ODPredictions): The object detection predictions. 
- conf_boxes (List[List[float]]): The conformalized bounding boxes. 
- verbose (bool): Whether to print the evaluation results. 

Returns 
------- 
- safety (torch.Tensor): The safety scores. 
- set_sizes (torch.Tensor): The set sizes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L871"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODClassificationConformalizer`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L884"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    loss='binary',
    prediction_set='lac',
    preprocess='softmax',
    backend='auto',
    guarantee_level='image',
    optimizer='binary_search',
    device='cpu',
    **kwargs
)
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1016"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    predictions: ODPredictions,
    alpha: float,
    bounds: List[float] = [0, 1],
    steps: int = 40,
    verbose: bool = True,
    overload_confidence_threshold: Optional[float] = None
) → Tensor
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1068"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(predictions: ODPredictions) → List
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1086"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    predictions: ODPredictions,
    parameters: Optional[ODParameters],
    conformalized_predictions: ODConformalizedPredictions,
    verbose: bool = True
) → Tuple[Tensor, Tensor]
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODConformalizer`
Class representing conformalizers for object detection tasks. 



**Attributes:**
 
---------- 
 - <b>`MULTIPLE_TESTING_CORRECTIONS`</b> (List[str]):  List of supported multiple testing correction methods. 
 - <b>`BACKENDS`</b> (List[str]):  List of supported backends. 
 - <b>`GUARANTEE_LEVELS`</b> (List[str]):  List of supported guarantee levels. 



**Args:**
 
---- 
 - <b>`backend`</b> (str):  The backend used for the conformalization. Only 'auto' is supported currently. 
 - <b>`guarantee_level`</b> (str):  The guarantee level for the conformalization. Must be one of ["image", "object"]. 
 - <b>`confidence_threshold`</b> (Optional[float]):  The confidence threshold used for objectness conformalization. Mutually exclusive with 'confidence_method', if set, then confidence_method must be None. 
 - <b>`multiple_testing_correction`</b> (Optional[str]):  The method used for multiple testing correction. Must be one of ["bonferroni"] or None. None implies no correction is applied, and that a List of Alphas is expected for calibration instead. 
 - <b>`confidence_method`</b> (Union[ConfidenceConformalizer, str, None]):  The method used for confidence conformalization. Mutually exclusive with 'confidence_threshold', if set, then confidence_threshold must be None. Either pass a ConfidenceConformalizer instance, a string representing the method (loss) name, or None. 
 - <b>`localization_method`</b> (Union[LocalizationConformalizer, str, None]):  The method used for localization conformalization. Either pass a LocalizationConformalizer instance, a string representing the method (loss) name, or None. 
 - <b>`classification_method`</b> (Union[ClassificationConformalizer, str, None]):  The method used for classification conformalization. Either pass a ClassificationConformalizer instance, a string representing the method (loss) name, or None. 
 - <b>`**kwargs`</b>:  Additional keyword arguments. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If the provided backend is not supported. 
 - <b>`ValueError`</b>:  If the provided guarantee level is not supported. 
 - <b>`ValueError`</b>:  If both confidence_threshold and confidence_method are provided. 
 - <b>`ValueError`</b>:  If neither confidence_threshold nor confidence_method are provided. 
 - <b>`ValueError`</b>:  If the provided multiple_testing_correction is not supported. 

Methods: 
------- calibrate(predictions, global_alpha, alpha_confidence, alpha_localization, alpha_classification, verbose=True)  Calibrates the conformalizers and returns the calibration results. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  The predictions to be calibrated. 
 - <b>`global_alpha`</b> (Optional[float]):  The global alpha value for calibration. If multiple_testing_correction is None, individual alpha values will be used for each conformalizer. 
 - <b>`alpha_confidence`</b> (Optional[float]):  The alpha value for the confidence conformalizer. 
 - <b>`alpha_localization`</b> (Optional[float]):  The alpha value for the localization conformalizer. 
 - <b>`alpha_classification`</b> (Optional[float]):  The alpha value for the classification conformalizer. 
 - <b>`verbose`</b> (bool, optional):  Whether to print calibration information. Defaults to True. 



**Returns:**
 
------- 
 - <b>`dict[str, Any]`</b>:  A dictionary containing the calibration results, including target alpha values and estimated lambda values for each conformalizer. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If the multiple_testing_correction is not provided or is not valid. 
 - <b>`ValueError`</b>:  If the global_alpha is not provided when using the Bonferroni multiple_testing_correction. 
 - <b>`ValueError`</b>:  If explicit alpha values are provided when using the Bonferroni multiple_testing_correction. 



**Note:**

> ---- - The multiple_testing_correction attribute of the class must be set before calling this method. - The conformalizers must be initialized before calling this method. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1183"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    backend: str = 'auto',
    guarantee_level: str = 'image',
    matching_function: str = 'hausdorff',
    confidence_threshold: Optional[float] = None,
    multiple_testing_correction: Optional[str] = None,
    confidence_method: Optional[ConfidenceConformalizer, str] = None,
    localization_method: Optional[LocalizationConformalizer, str] = None,
    localization_prediction_set: str = 'additive',
    classification_method: Optional[ClassificationConformalizer, str] = None,
    classification_prediction_set: str = 'lac',
    device='cpu',
    **kwargs
)
```

Initialize the ODClassificationConformalizer object. 

Parameters 
---------- 
- backend (str): The backend used for the conformalization. Only 'auto' is supported currently. 
- guarantee_level (str): The guarantee level for the conformalization. Must be one of ["image", "object"]. 
- confidence_threshold (Optional[float]): The confidence threshold used for objectness conformalization.  Mutually exclusive with 'confidence_method', if set, then confidence_method must be None. 
- multiple_testing_correction (str): The method used for multiple testing correction. Must be one of ["bonferroni"] or None. None implies no correction is applied, and that a List of Alphas is expected for calibration instead. 
- confidence_method (Union[ConfidenceConformalizer, str, None]): The method used for confidence conformalization. Mutually exclusive with 'confidence_threshold', if set, then confidence_threshold must be None. Either pass a ConfidenceConformalizer instance, a string representing the method (loss) name, or None. 
- localization_method (Union[LocalizationConformalizer, str, None]): The method used for localization conformalization. Either pass a LocalizationConformalizer instance, a string representing the method (loss) name, or None. 
- classification_method (Union[ClassificationConformalizer, str, None]): The method used for classification conformalization. Either pass a ClassificationConformalizer instance, a string representing the method (loss) name, or None. 
- kwargs: Additional keyword arguments. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1341"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    predictions: ODPredictions,
    global_alpha: Optional[float] = None,
    alpha_confidence: Optional[float] = None,
    alpha_localization: Optional[float] = None,
    alpha_classification: Optional[float] = None,
    verbose: bool = True
) → ODParameters
```

Calibrates the conformalizers and returns the calibration results. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  The predictions to be calibrated. 
 - <b>`global_alpha`</b> (Optional[float]):  The global alpha value for calibration. If multiple_testing_correction is None, individual alpha values will be used for each conformalizer. 
 - <b>`alpha_confidence`</b> (Optional[float]):  The alpha value for the confidence conformalizer. 
 - <b>`alpha_localization`</b> (Optional[float]):  The alpha value for the localization conformalizer. 
 - <b>`alpha_classification`</b> (Optional[float]):  The alpha value for the classification conformalizer. 
 - <b>`verbose`</b> (bool, optional):  Whether to print calibration information. Defaults to True. 



**Returns:**
 
------- 
 - <b>`dict[str, Any]`</b>:  A dictionary containing the calibration results, including target alpha values and estimated lambda values for each conformalizer. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If the multiple_testing_correction is not provided or is not valid. 
 - <b>`ValueError`</b>:  If the global_alpha is not provided when using the Bonferroni multiple_testing_correction. 
 - <b>`ValueError`</b>:  If explicit alpha values are provided when using the Bonferroni multiple_testing_correction. 



**Note:**

> ---- - The multiple_testing_correction attribute of the class must be set before calling this method. - The conformalizers must be initialized before calling this method. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1557"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(
    predictions: ODPredictions,
    parameters: Optional[ODParameters] = None,
    verbose: bool = True
) → ODConformalizedPredictions
```

Conformalize the given predictions. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  The predictions to be conformalized. 
 - <b>`parameters`</b> (Optional[ODParameters]):  The parameters to be used for conformalization. If None, the last parameters will be used.results 
 - <b>`verbose`</b> (bool):  Whether to print conformalization information. 



**Returns:**
 
------- 
 - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  A tuple containing the conformalized predictions. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1647"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1765"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AsymptoticLocalizationObjectnessConformalizer`
A class that performs risk conformalization for object detection predictions with asymptotic localization and objectness. 



**Args:**
 
---- 
 - <b>`prediction_set`</b> (str):  The type of prediction set to use. Must be one of "additive", "multiplicative", or "adaptative". 
 - <b>`localization_loss`</b> (str):  The type of localization loss to use. Must be one of "pixelwise" or "boxwise". 
 - <b>`optimizer`</b> (str):  The type of optimizer to use. Must be one of "gaussianprocess", "gpr", "kriging", "mc", or "montecarlo". 



**Attributes:**
 
---------- 
 - <b>`ACCEPTED_LOSSES`</b> (dict):  A dictionary mapping accepted localization losses to their corresponding classes. 
 - <b>`loss_name`</b> (str):  The name of the localization loss. 
 - <b>`loss`</b> (Loss):  An instance of the localization loss class. 
 - <b>`prediction_set`</b> (str):  The type of prediction set. 
 - <b>`lbd`</b> (tuple):  The calibrated lambda values. 

Methods: 
------- 
 - <b>`_get_risk_function`</b>:  Returns the risk function for optimization. 
 - <b>`_correct_risk`</b>:  Corrects the risk using the number of predictions and the upper bound of the loss. 
 - <b>`calibrate`</b>:  Calibrates the conformalizer using the given predictions. 
 - <b>`conformalize`</b>:  Conformalizes the predictions using the calibrated lambda values. 
 - <b>`evaluate`</b>:  Evaluates the conformalized predictions. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1797"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    prediction_set: str = 'additive',
    localization_loss: str = 'boxwise',
    optimizer: str = 'gpr'
)
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1888"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    predictions: ODPredictions,
    alpha: float = 0.1,
    steps: int = 13,
    bounds: list = [(0, 500), (0.0, 1.0)],
    verbose: bool = True
)
```

Calibrates the conformalizer using the given predictions. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  The object detection predictions. 
 - <b>`alpha`</b> (float):  The significance level. 
 - <b>`steps`</b> (int):  The number of optimization steps. 
 - <b>`bounds`</b> (list):  The bounds for the optimization variables. 
 - <b>`verbose`</b> (bool):  Whether to print verbose output. 



**Returns:**
 
------- 
 - <b>`tuple`</b>:  The calibrated lambda values. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If the conformalizer has already been calibrated. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1932"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(predictions: ODPredictions)
```

Conformalizes the predictions using the calibrated lambda values. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  The object detection predictions. 



**Returns:**
 
------- 
 - <b>`list`</b>:  The conformalized bounding boxes. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If the conformalizer has not been calibrated. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/cp.py#L1961"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(predictions: ODPredictions, conf_boxes: list, verbose: bool = True)
```

Evaluates the conformalized predictions. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  The object detection predictions. 
 - <b>`conf_boxes`</b> (list):  The conformalized bounding boxes. 
 - <b>`verbose`</b> (bool):  Whether to print verbose output. 



**Returns:**
 
------- 
 - <b>`tuple`</b>:  The evaluation results. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If the conformalizer has not been calibrated or the predictions have not been conformalized. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
