<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `base.tr`
Base training and tolerance region utilities for conformal prediction. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `hoeffding`

```python
hoeffding(Rhat, n, delta)
```

Hoeffding's inequality bound for binomial proportion confidence intervals. 



**Args:**
 
---- 
 - <b>`Rhat`</b>:  Empirical risk estimate. 
 - <b>`n`</b>:  Number of samples. 
 - <b>`delta`</b>:  Confidence level. 



**Returns:**
 
------- Bound on the risk with probability at least 1 - delta. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bernstein_emp`

```python
bernstein_emp(Rhat, n, delta)
```

Empirical Bernstein bound for binomial proportion confidence intervals. 



**Args:**
 
---- 
 - <b>`Rhat`</b>:  Empirical risk estimate. 
 - <b>`n`</b>:  Number of samples. 
 - <b>`delta`</b>:  Confidence level. 



**Returns:**
 
------- Bound on the risk with probability at least 1 - delta. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bernstein`

```python
bernstein(Rhat, n, delta)
```

Bernstein's inequality bound for binomial proportion confidence intervals. 



**Args:**
 
---- 
 - <b>`Rhat`</b>:  Empirical risk estimate. 
 - <b>`n`</b>:  Number of samples. 
 - <b>`delta`</b>:  Confidence level. 



**Returns:**
 
------- Bound on the risk with probability at least 1 - delta. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bernstein_uni`

```python
bernstein_uni(Rhat, n, delta)
```

Uniform Bernstein bound for binomial proportion confidence intervals. 



**Args:**
 
---- 
 - <b>`Rhat`</b>:  Empirical risk estimate. 
 - <b>`n`</b>:  Number of samples. 
 - <b>`delta`</b>:  Confidence level. 



**Returns:**
 
------- Bound on the risk with probability at least 1 - delta. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bernstein_uni_lim`

```python
bernstein_uni_lim(Rhat, n, delta)
```

Limited uniform Bernstein bound for binomial proportion confidence intervals. 



**Args:**
 
---- 
 - <b>`Rhat`</b>:  Empirical risk estimate. 
 - <b>`n`</b>:  Number of samples. 
 - <b>`delta`</b>:  Confidence level. 



**Returns:**
 
------- Bound on the risk with probability at least 1 - delta. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `binom_inv_cdf`

```python
binom_inv_cdf(Rhat, n, delta, device='cpu')
```

Inverse binomial CDF for confidence interval calculation. 



**Args:**
 
---- 
 - <b>`Rhat`</b>:  Empirical risk estimate. 
 - <b>`n`</b>:  Number of samples. 
 - <b>`delta`</b>:  Confidence level. 
 - <b>`device`</b>:  Torch device string. 



**Returns:**
 
------- Upper bound on the risk with probability at least 1 - delta. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ToleranceRegion`
Abstract base class for tolerance region conformal predictors. 

Provides interface for calibrating, conformalizing, and evaluating predictions using different inequalities and optimizers. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    inequality: Union[str, Callable] = 'binomial_inverse_cdf',
    optimizer: str = 'binary_search',
    optimizer_args: dict = None
)
```

Initialize a ToleranceRegion instance. 



**Args:**
 
---- 
 - <b>`inequality`</b> (Union[str, Callable]):  Inequality function or name. 
 - <b>`optimizer`</b> (str):  Optimizer name. 
 - <b>`optimizer_args`</b> (dict):  Arguments for the optimizer. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If the inequality or optimizer is not recognized. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    preds: Predictions,
    alpha: float = 0.1,
    delta: float = 0.1,
    verbose: bool = True,
    **kwargs
)
```

Calibrate the tolerance region on the given predictions. 



**Args:**
 
---- 
 - <b>`preds`</b> (Predictions):  Predictions to calibrate on. 
 - <b>`alpha`</b> (float):  Miscoverage level. 
 - <b>`delta`</b> (float):  Confidence level. 
 - <b>`verbose`</b> (bool):  Whether to print detailed output. 
 - <b>`**kwargs`</b>:  Additional arguments. 



**Raises:**
 
------ 
 - <b>`NotImplementedError`</b>:  If not implemented in subclass. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L229"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds: Predictions, verbose: bool = True, **kwargs)
```

Conformalize the given predictions. 



**Args:**
 
---- 
 - <b>`preds`</b> (Predictions):  Predictions to conformalize. 
 - <b>`verbose`</b> (bool):  Whether to print detailed output. 
 - <b>`**kwargs`</b>:  Additional arguments. 



**Raises:**
 
------ 
 - <b>`NotImplementedError`</b>:  If not implemented in subclass. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(preds: Predictions, verbose: bool = True, **kwargs)
```

Evaluate the tolerance region on the given predictions. 



**Args:**
 
---- 
 - <b>`preds`</b> (Predictions):  Predictions to evaluate. 
 - <b>`verbose`</b> (bool):  Whether to print detailed output. 
 - <b>`**kwargs`</b>:  Additional arguments. 



**Raises:**
 
------ 
 - <b>`NotImplementedError`</b>:  If not implemented in subclass. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L266"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CombiningToleranceRegions`
Combine multiple tolerance regions, e.g., using Bonferroni correction. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L269"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*tregions: ToleranceRegion, mode: str = 'bonferroni')
```

Initialize a CombiningToleranceRegions instance. 



**Args:**
 
---- 
 - <b>`*tregions (ToleranceRegion)`</b>:  Tolerance region instances to combine. 
 - <b>`mode`</b> (str):  Combination mode (default: 'bonferroni'). 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L281"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `calibrate`

```python
calibrate(
    preds: Predictions,
    alpha: float = 0.1,
    delta: float = 0.1,
    parameters: Optional[list] = None
)
```

Calibrate all tolerance regions with appropriate correction. 



**Args:**
 
---- 
 - <b>`preds`</b> (Predictions):  Predictions to calibrate on. 
 - <b>`alpha`</b> (float):  Miscoverage level. 
 - <b>`delta`</b> (float):  Confidence level. 
 - <b>`parameters`</b> (Optional[list]):  List of parameter dicts for each region. 



**Returns:**
 
------- 
 - <b>`list`</b>:  Calibration results for each region. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conformalize`

```python
conformalize(preds)
```

Conformalize predictions using all tolerance regions. 



**Args:**
 
---- 
 - <b>`preds`</b>:  Predictions to conformalize. 



**Returns:**
 
------- 
 - <b>`list`</b>:  Conformalized predictions for each region. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/tr.py#L329"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(preds, verbose=True)
```

Evaluate all tolerance regions on the given predictions. 



**Args:**
 
---- 
 - <b>`preds`</b>:  Predictions to evaluate. 
 - <b>`verbose`</b> (bool):  Whether to print detailed output. 



**Returns:**
 
------- 
 - <b>`list`</b>:  Evaluation results for each region. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
