<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `base.optim`
Optimizers for conformal prediction and related search procedures. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Optimizer`
Abstract base class for optimizers used in conformal prediction calibration. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `optimize`

```python
optimize(objective_function: Callable, alpha: float, **kwargs) → float
```

Optimize the objective function to satisfy the risk constraint. 



**Args:**
 
---- 
 - <b>`objective_function`</b> (Callable):  The function to optimize. 
 - <b>`alpha`</b> (float):  The risk threshold. 
 - <b>`**kwargs`</b>:  Additional arguments for the optimizer. 



**Returns:**
 
------- 
 - <b>`float`</b>:  The optimal parameter value. 



**Raises:**
 
------ 
 - <b>`NotImplementedError`</b>:  If not implemented in subclass. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BinarySearchOptimizer`
Optimizer using binary search in 1D (or multi-D) for risk calibration. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the BinarySearchOptimizer. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `optimize`

```python
optimize(
    objective_function: Callable,
    alpha: float,
    bounds: Union[Tuple, List, List[Tuple]],
    steps: int,
    epsilon=1e-05,
    verbose=True
) → float
```

Perform binary search to find the optimal parameter. 



**Args:**
 
---- 
 - <b>`objective_function`</b> (Callable):  Function of one parameter lbd (use partials), which includes the correction part. 
 - <b>`alpha`</b> (float):  Risk threshold. 
 - <b>`bounds`</b> (Union[Tuple, List, List[Tuple]]):  Search bounds. 
 - <b>`steps`</b> (int):  Number of search steps. 
 - <b>`epsilon`</b> (float):  Precision threshold. 
 - <b>`verbose`</b> (bool):  Whether to print progress. 



**Returns:**
 
------- 
 - <b>`float`</b>:  The optimal parameter value. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GaussianProcessOptimizer`
Optimizer using Gaussian Process Regression (Bayesian optimization) for risk calibration. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the GaussianProcessOptimizer. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `optimize`

```python
optimize(
    objective_function: Callable,
    alpha: float,
    bounds: Union[Tuple, List, List[Tuple]],
    steps: int,
    epsilon=1e-05,
    verbose=True
) → Union[float, ndarray, NoneType]
```

Optimize using Gaussian Process Regression (Bayesian optimization). 



**Args:**
 
---- 
 - <b>`objective_function`</b> (Callable):  The function to optimize. 
 - <b>`alpha`</b> (float):  The risk threshold. 
 - <b>`bounds`</b> (Union[Tuple, List, List[Tuple]]):  Search bounds. 
 - <b>`steps`</b> (int):  Number of optimization steps. 
 - <b>`epsilon`</b> (float):  Precision threshold. 
 - <b>`verbose`</b> (bool):  Whether to print progress. 



**Returns:**
 
------- 
 - <b>`float or np.ndarray or None`</b>:  The optimal parameter value, or None if not found. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L186"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MonteCarloOptimizer`
Optimizer using Monte Carlo random search for risk calibration. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the MonteCarloOptimizer. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `optimize`

```python
optimize(
    objective_function: Callable,
    alpha: float,
    bounds: Union[Tuple, List[Tuple]],
    steps: int,
    epsilon=0.0001,
    verbose=True
) → Union[float, ndarray, NoneType]
```

Optimize using Monte Carlo random search. 



**Args:**
 
---- 
 - <b>`objective_function`</b> (Callable):  The function to optimize. 
 - <b>`alpha`</b> (float):  The risk threshold. 
 - <b>`bounds`</b> (Union[Tuple, List[Tuple]]):  Search bounds. 
 - <b>`steps`</b> (int):  Number of optimization steps. 
 - <b>`epsilon`</b> (float):  Precision threshold. 
 - <b>`verbose`</b> (bool):  Whether to print progress. 



**Returns:**
 
------- 
 - <b>`float or np.ndarray or None`</b>:  The optimal parameter value, or None if not found. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
