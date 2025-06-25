<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `base.cp`
Base module for conformal prediction classes. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Conformalizer`
Abstract base class for conformal prediction methods. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the Conformalizer base class. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(preds, alpha=0.1)
```

Calibrate the conformal predictor on the given predictions. 



**Args:**
 
---- 
 - <b>`preds`</b>:  Predictions to calibrate on. 
 - <b>`alpha`</b> (float):  Miscoverage level (default: 0.1). 



**Raises:**
 
------ 
 - <b>`NotImplementedError`</b>:  If not implemented in subclass. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds)
```

Conformalize the given predictions. 



**Args:**
 
---- 
 - <b>`preds`</b>:  Predictions to conformalize. 



**Raises:**
 
------ 
 - <b>`NotImplementedError`</b>:  If not implemented in subclass. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(preds, verbose=True)
```

Evaluate the conformal predictor on the given predictions. 



**Args:**
 
---- 
 - <b>`preds`</b>:  Predictions to evaluate. 
 - <b>`verbose`</b> (bool):  Whether to print detailed output (default: True). 



**Raises:**
 
------ 
 - <b>`NotImplementedError`</b>:  If not implemented in subclass. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
