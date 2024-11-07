<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `base.tr`





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `hoeffding`

```python
hoeffding(Rhat, n, delta)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bernstein_emp`

```python
bernstein_emp(Rhat, n, delta)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bernstein`

```python
bernstein(Rhat, n, delta)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bernstein_uni`

```python
bernstein_uni(Rhat, n, delta)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bernstein_uni_lim`

```python
bernstein_uni_lim(Rhat, n, delta)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `binom_inv_cdf`

```python
binom_inv_cdf(Rhat, n, delta)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ToleranceRegion`




<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    inequality: Union[str, Callable] = 'binomial_inverse_cdf',
    optimizer: str = 'binary_search',
    optimizer_args: dict = {}
)
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(preds, alpha=0.1, delta=0.1, verbose=True, **kwargs)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds, verbose=True, **kwargs)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(preds, verbose=True, **kwargs)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CombiningToleranceRegions`




<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*tregions: ToleranceRegion, mode: str = 'bonferroni')
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(preds, alpha=0.1, delta=0.1, parameters=None)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(preds, verbose=True)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
