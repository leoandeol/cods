<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/loss.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `classif.loss`
Loss functions for conformal classification. 

**Global Variables**
---------------
- **CLASSIFICATION_LOSSES**


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/loss.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassificationLoss`
Abstract base class for classification loss functions. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/loss.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(upper_bound: float = 1, **kwargs)
```

Initialize the ClassificationLoss base class. 



**Args:**
 
---- 
 - <b>`upper_bound`</b> (float, optional):  Upper bound for the loss. Defaults to 1. 
 - <b>`**kwargs`</b>:  Additional arguments. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/loss.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_cls: Tensor, lbd: float)
```

Get the conformal prediction set for a given threshold. 



**Args:**
 
---- 
 - <b>`pred_cls`</b> (torch.Tensor):  Predicted class probabilities. 
 - <b>`lbd`</b> (float):  Threshold parameter. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Indices of the conformal prediction set. 



**Raises:**
 
------ 
 - <b>`NotImplementedError`</b>:  If not implemented in subclass. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/loss.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LACLoss`
Loss function for Least Ambiguous Conformal (LAC) prediction sets. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/loss.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(upper_bound: float = 1, **kwargs)
```

Initialize the LACLoss class. 



**Args:**
 
---- 
 - <b>`upper_bound`</b> (float, optional):  Upper bound for the loss. Defaults to 1. 
 - <b>`**kwargs`</b>:  Additional arguments. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/loss.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_cls: Tensor, lbd: float)
```

Get the conformal prediction set for a given threshold. 



**Args:**
 
---- 
 - <b>`pred_cls`</b> (torch.Tensor):  Predicted class probabilities. 
 - <b>`lbd`</b> (float):  Threshold parameter. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Indices of the conformal prediction set. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
