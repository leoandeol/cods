<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `base.cp`






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L1"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Conformalizer`




<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L2"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(preds, alpha=0.1)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(preds, verbose=True)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RiskConformalizer`




<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(preds, alpha=0.1)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate_multidim`

```python
calibrate_multidim(preds, alpha=0.1)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(preds, verbose=True)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CombiningConformalPredictionSets`




<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*conformalizers, mode='bonferroni')
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(preds, alpha=0.1, parameters=None)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/cp.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(preds, verbose=True)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
