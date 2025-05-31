<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.optim`






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FirstStepMonotonizingOptimizer`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `optimize`

```python
optimize(
    predictions: ODPredictions,
    confidence_loss: ODLoss,
    localization_loss: ODLoss,
    classification_loss: ODLoss,
    matching_function,
    alpha: float,
    device: str,
    B: float = 1,
    bounds: List[float] = [0, 1],
    init_lambda: float = 1,
    verbose: bool = False
)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L544"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SecondStepMonotonizingOptimizer`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L545"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L569"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate_risk`

```python
evaluate_risk(
    lbd,
    loss,
    final_lbd_conf,
    predictions,
    build_predictions,
    matching_function
)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L756"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `optimize`

```python
optimize(
    predictions: ODPredictions,
    build_predictions,
    loss: ODLoss,
    matching_function,
    alpha: float,
    device: str,
    B: float = 1,
    bounds: List[float] = [0, 1],
    steps=13,
    epsilon=1e-10,
    verbose: bool = False
)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
