<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/tr.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `classif.tr`
Tolerance region implementation for conformal classification. 

**Global Variables**
---------------
- **CLASSIFICATION_LOSSES**


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/tr.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassificationToleranceRegion`
Tolerance region for conformal classification tasks. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/tr.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    loss='lac',
    inequality='binomial_inverse_cdf',
    optimizer='binary_search',
    preprocess='softmax',
    device='cpu',
    optimizer_args=None
)
```

Initialize the ClassificationToleranceRegion. 



**Args:**
 
---- 
 - <b>`loss`</b> (str or ClassificationLoss):  Loss function or its name. 
 - <b>`inequality`</b> (str):  Inequality function name. 
 - <b>`optimizer`</b> (str):  Optimizer name. 
 - <b>`preprocess`</b> (str or Callable):  Preprocessing function or its name. 
 - <b>`device`</b> (str):  Device to use. 
 - <b>`optimizer_args`</b> (dict, optional):  Arguments for the optimizer. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/tr.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    predictions: ClassificationPredictions,
    alpha=0.1,
    delta=0.1,
    steps=13,
    bounds=None,
    verbose=True,
    objectness_threshold=0.8
)
```

Calibrate the tolerance region for conformal classification. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ClassificationPredictions):  Predictions to calibrate on. 
 - <b>`alpha`</b> (float, optional):  Miscoverage level. Defaults to 0.1. 
 - <b>`delta`</b> (float, optional):  Confidence level. Defaults to 0.1. 
 - <b>`steps`</b> (int, optional):  Number of optimization steps. Defaults to 13. 
 - <b>`bounds`</b> (list, optional):  Search bounds. Defaults to [0, 1]. 
 - <b>`verbose`</b> (bool, optional):  Whether to print progress. Defaults to True. 
 - <b>`objectness_threshold`</b> (float, optional):  Objectness threshold. Defaults to 0.8. 



**Returns:**
 
------- 
 - <b>`float`</b>:  The calibrated lambda value. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/tr.py#L202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(
    predictions: ClassificationPredictions,
    verbose: bool = True,
    **kwargs
) → list
```

Conformalize the predictions using the calibrated lambda. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ClassificationPredictions):  Predictions to conformalize. 
 - <b>`verbose`</b> (bool, optional):  Whether to print progress. Defaults to True. 
 - <b>`**kwargs`</b>:  Additional arguments. 



**Returns:**
 
------- 
 - <b>`list`</b>:  List of conformalized prediction sets. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/tr.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    preds: ClassificationPredictions,
    conf_cls: list,
    verbose=True,
    **kwargs
)
```

Evaluate the conformalized predictions. 



**Args:**
 
---- 
 - <b>`preds`</b> (ClassificationPredictions):  Predictions to evaluate. 
 - <b>`conf_cls`</b> (list):  Conformalized prediction sets. 
 - <b>`verbose`</b> (bool, optional):  Whether to print progress. Defaults to True. 
 - <b>`**kwargs`</b>:  Additional arguments. 



**Returns:**
 
------- 
 - <b>`tuple`</b>:  Tuple of (coverage, average set size). 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
