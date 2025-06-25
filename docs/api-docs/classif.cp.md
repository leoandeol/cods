<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/cp.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `classif.cp`
Conformalizer for conformal classification tasks. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/cp.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassificationConformalizer`
Implements conformal prediction for classification using various non-conformity scores. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/cp.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(method='lac', preprocess='softmax', device='cpu')
```

Initialize the ClassificationConformalizer. 



**Args:**
 
---- 
 - <b>`method`</b> (str or ClassifNCScore):  Non-conformity score method or instance. 
 - <b>`preprocess`</b> (str):  Preprocessing function name. 
 - <b>`device`</b> (str):  Device to use. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If method or preprocess is not accepted. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/cp.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    preds: ClassificationPredictions,
    alpha: float = 0.1,
    verbose: bool = True,
    lbd_minus: bool = False
) → Tuple[Tensor, Tensor]
```

Calibrate the conformalizer and compute the quantile for the non-conformity scores. 



**Args:**
 
---- 
 - <b>`preds`</b> (ClassificationPredictions):  Predictions to calibrate on. 
 - <b>`alpha`</b> (float, optional):  Miscoverage level. Defaults to 0.1. 
 - <b>`verbose`</b> (bool, optional):  Whether to print progress. Defaults to True. 
 - <b>`lbd_minus`</b> (bool, optional):  Whether to use the minus quantile. Defaults to False. 



**Returns:**
 
------- 
 - <b>`Tuple[torch.Tensor, torch.Tensor]`</b>:  The calibrated quantile and the non-conformity scores. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/cp.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds: ClassificationPredictions) → list
```

Conformalize the predictions using the calibrated quantile. 



**Args:**
 
---- 
 - <b>`preds`</b> (ClassificationPredictions):  Predictions to conformalize. 



**Returns:**
 
------- 
 - <b>`list`</b>:  List of conformalized prediction sets. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If the conformalizer is not calibrated. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/cp.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(preds: ClassificationPredictions, conf_cls: list, verbose=True)
```

Evaluate the conformalized predictions. 



**Args:**
 
---- 
 - <b>`preds`</b> (ClassificationPredictions):  Predictions to evaluate. 
 - <b>`conf_cls`</b> (list):  Conformalized prediction sets. 
 - <b>`verbose`</b> (bool, optional):  Whether to print progress. Defaults to True. 



**Returns:**
 
------- 
 - <b>`tuple`</b>:  Tuple of (coverage, average set size). 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If the conformalizer is not calibrated or predictions are not conformalized. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
