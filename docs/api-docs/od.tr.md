<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.tr`






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LocalizationToleranceRegion`
Tolerance region for object localization tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    prediction_set: str = 'additive',
    loss: Optional[str] = None,
    optimizer: str = 'binary_search',
    inequality: Union[str, Callable] = 'binomial_inverse_cdf'
)
```

Initialize the LocalizationToleranceRegion. 



**Args:**
 
 - <b>`prediction_set`</b> (str):  The type of prediction set to use. Must be one of "additive", "multiplicative", or "adaptative". 
 - <b>`loss`</b> (str, None):  The type of loss to use. Must be one of "pixelwise" or "boxwise". 
 - <b>`optimizer`</b> (str):  The type of optimizer to use. Must be one of "binary_search", "gaussianprocess", "gpr", or "kriging". 
 - <b>`inequality`</b> (str, Callable):  The type of inequality function to use. Must be one of "binomial_inverse_cdf" or a custom callable. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the loss or optimizer is not accepted. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    preds: ODPredictions,
    alpha: float = 0.1,
    delta: float = 0.1,
    steps: int = 13,
    bounds: Tuple[float, float] = (0, 1),
    verbose: bool = True,
    confidence_threshold: Optional[float] = None
)
```

Calibrate the tolerance region. 



**Args:**
 
 - <b>`preds`</b> (ODPredictions):  The object detection predictions. 
 - <b>`alpha`</b> (float):  The significance level. 
 - <b>`delta`</b> (float):  The confidence level. 
 - <b>`steps`</b> (int):  The number of steps for optimization. 
 - <b>`bounds`</b> (Tuple[float, float]):  The bounds for optimization. 
 - <b>`verbose`</b> (bool):  Whether to print verbose output. 
 - <b>`confidence_threshold`</b> (float, None):  The confidence threshold. 



**Returns:**
 
 - <b>`float`</b>:  The calibrated lambda value. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds: ODPredictions)
```

Conformalize the object detection predictions. 



**Args:**
 
 - <b>`preds`</b> (ODPredictions):  The object detection predictions. 



**Returns:**
 
 - <b>`list`</b>:  The conformalized bounding boxes. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(preds: ODPredictions, conf_boxes: list, verbose=True)
```

Evaluate the conformalized object detection predictions. 



**Args:**
 
 - <b>`preds`</b> (ODPredictions):  The object detection predictions. 
 - <b>`conf_boxes`</b> (list):  The conformalized bounding boxes. 
 - <b>`verbose`</b> (bool):  Whether to print verbose output. 



**Returns:**
 
 - <b>`Tuple[float, torch.Tensor]`</b>:  The safety and set sizes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L255"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConfidenceToleranceRegion`
Tolerance region for object confidence tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L262"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    loss: str = 'box_number',
    inequality: str = 'binomial_inverse_cdf',
    optimizer: str = 'binary_search'
) → None
```

Initialize the ConfidenceToleranceRegion. 



**Args:**
 
 - <b>`loss`</b> (str):  The type of loss to use. Must be one of "box_number". 
 - <b>`inequality`</b> (str):  The type of inequality function to use. Must be one of "binomial_inverse_cdf" or a custom callable. 
 - <b>`optimizer`</b> (str):  The type of optimizer to use. Must be one of "binary_search", "gaussianprocess", "gpr", or "kriging". 



**Raises:**
 
 - <b>`ValueError`</b>:  If the loss or optimizer is not accepted. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L294"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    preds: ODPredictions,
    alpha: float = 0.1,
    delta: float = 0.1,
    steps: int = 13,
    bounds: Tuple[float, float] = (0, 1),
    verbose: bool = True,
    confidence_threshold: Optional[float] = None
) → float
```

Calibrate the tolerance region. 



**Args:**
 
 - <b>`preds`</b> (ODPredictions):  The object detection predictions. 
 - <b>`alpha`</b> (float):  The significance level. 
 - <b>`delta`</b> (float):  The confidence level. 
 - <b>`steps`</b> (int):  The number of steps for optimization. 
 - <b>`bounds`</b> (Tuple[float, float]):  The bounds for optimization. 
 - <b>`verbose`</b> (bool):  Whether to print verbose output. 
 - <b>`confidence_threshold`</b> (float, None):  The confidence threshold. 



**Returns:**
 
 - <b>`float`</b>:  The calibrated lambda value. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L403"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds: ODPredictions, verbose: bool = True, **kwargs) → float
```

Conformalize the predictions. 



**Args:**
 
 - <b>`preds`</b> (ODPredictions):  The object detection predictions. 
 - <b>`verbose`</b> (bool):  Whether to print verbose output. 



**Returns:**
 
 - <b>`float`</b>:  The confidence threshold. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L421"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    preds: ODPredictions,
    conf_boxes: list,
    verbose: bool = True,
    **kwargs
) → Tuple[Tensor, Tensor]
```

Evaluate the tolerance region. 



**Args:**
 
 - <b>`preds`</b> (ODPredictions):  The object detection predictions. 
 - <b>`conf_boxes`</b> (list):  The confidence boxes. 
 - <b>`verbose`</b> (bool):  Whether to print verbose output. 



**Returns:**
 
 - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  The coverage and set sizes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L459"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODToleranceRegion`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L462"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    localization_loss: Union[NoneType, str, LocalizationToleranceRegion] = 'boxwise',
    confidence_loss: Union[NoneType, str, ConfidenceToleranceRegion] = 'box_number',
    classification_loss: Union[NoneType, str, ClassificationToleranceRegion] = 'lac',
    inequality='binomial_inverse_cdf',
    margins=1,
    prediction_set='additive',
    multiple_testing_correction: str = 'bonferroni',
    confidence_threshold: Optional[float] = None,
    **kwargs
)
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L528"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(preds, alpha=0.1, delta=0.1, steps=13, bounds=[0, 1], verbose=True)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L610"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds: ODPredictions)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/tr.py#L627"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(preds: ODPredictions, conf_boxes, conf_cls, verbose=True)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
