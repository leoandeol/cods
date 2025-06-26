<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/score.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `classif.score`
Non-conformity scores for conformal classification. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/score.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassifNCScore`
Abstract base class for classification non-conformity score functions. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/score.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_classes, **kwargs)
```

Initialize the ClassifNCScore base class. 



**Args:**
 
---- 
 - <b>`n_classes`</b> (int):  Number of classes. 
 - <b>`**kwargs`</b>:  Additional arguments. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/score.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_cls, quantile)
```

Get the conformal prediction set for a given quantile. 



**Args:**
 
---- 
 - <b>`pred_cls`</b> (torch.Tensor):  Predicted class probabilities. 
 - <b>`quantile`</b> (float):  Quantile threshold. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Indices of the conformal prediction set. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/score.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LACNCScore`
Non-conformity score for Least Ambiguous Conformal (LAC) prediction sets. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/score.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_classes: int, **kwargs)
```

Initialize the LACNCScore class. 



**Args:**
 
---- 
 - <b>`n_classes`</b> (int):  Number of classes. 
 - <b>`**kwargs`</b>:  Additional arguments. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/score.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_cls, quantile)
```

Get the conformal prediction set for a given quantile. 



**Args:**
 
---- 
 - <b>`pred_cls`</b> (torch.Tensor):  Predicted class probabilities. 
 - <b>`quantile`</b> (float):  Quantile threshold. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Indices of the conformal prediction set. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/score.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `APSNCScore`
Non-conformity score for Adaptive Prediction Sets (APS). 

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/score.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_classes, **kwargs)
```

Initialize the APSNCScore class. 



**Args:**
 
---- 
 - <b>`n_classes`</b> (int):  Number of classes. 
 - <b>`**kwargs`</b>:  Additional arguments. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/score.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_cls: Tensor, quantile: float)
```

Get the conformal prediction set for a given quantile. 



**Args:**
 
---- 
 - <b>`pred_cls`</b> (torch.Tensor):  Predicted class probabilities. 
 - <b>`quantile`</b> (float):  Quantile threshold. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Indices of the conformal prediction set. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
