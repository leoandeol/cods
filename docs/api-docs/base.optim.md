<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `base.optim`






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Optimizer`







---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `optimize`

```python
optimize(objective_function: Callable, alpha: float, **kwargs) → float
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BinarySearchOptimizer`




<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

params: epsilon: objective_function: function of one parameter lbd (use partials), which includes the correction part 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GaussianProcessOptimizer`




<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MonteCarloOptimizer`




<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/optim.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `optimize`

```python
optimize(
    objective_function: Callable,
    alpha: float,
    bounds: Union[Tuple, List[Tuple]],
    steps: int,
    epsilon=0.0001,
    verbose=True
) → float
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
