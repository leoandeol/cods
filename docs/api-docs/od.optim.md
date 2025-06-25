<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.optim`
Optimizers for conformal object detection calibration and risk control. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FirstStepMonotonizingOptimizer`
Optimizer for the first step of monotonic risk control in object detection. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the FirstStepMonotonizingOptimizer. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    bounds: List[float] = None,
    init_lambda: float = 1,
    verbose: bool = False
)
```

Optimize the risk for object detection using monotonic risk control. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`confidence_loss`</b> (ODLoss):  Loss function for confidence. 
 - <b>`localization_loss`</b> (ODLoss):  Loss function for localization. 
 - <b>`classification_loss`</b> (ODLoss):  Loss function for classification. 
 - <b>`matching_function`</b>:  Function to match predictions to ground truth. 
 - <b>`alpha`</b> (float):  Risk threshold. 
 - <b>`device`</b> (str):  Device to use. 
 - <b>`B`</b> (float, optional):  Upper bound. Defaults to 1. 
 - <b>`bounds`</b> (list, optional):  Search bounds. Defaults to [0, 1]. 
 - <b>`init_lambda`</b> (float, optional):  Initial lambda value. Defaults to 1. 
 - <b>`verbose`</b> (bool, optional):  Whether to print progress. Defaults to False. 



**Returns:**
 
------- 
 - <b>`float`</b>:  The optimal lambda value. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If no solution is found satisfying the constraints. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L571"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SecondStepMonotonizingOptimizer`
Optimizer for the second step of monotonic risk control in object detection. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L574"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the SecondStepMonotonizingOptimizer. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L599"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Evaluate the risk for a given lambda value. 



**Args:**
 
---- 
 - <b>`lbd`</b>:  Lambda value. 
 - <b>`loss`</b>:  Loss function. 
 - <b>`final_lbd_conf`</b>:  Final lambda for confidence. 
 - <b>`predictions`</b>:  Object detection predictions. 
 - <b>`build_predictions`</b>:  Function to build predictions. 
 - <b>`matching_function`</b>:  Function to match predictions to ground truth. 



**Returns:**
 
------- The evaluated risk. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/optim.py#L797"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    bounds: List[float] = None,
    steps=13,
    epsilon=1e-10,
    verbose: bool = False
)
```

Optimize the risk for object detection using monotonic risk control (second step). 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Object detection predictions. 
 - <b>`build_predictions`</b>:  Function to build predictions. 
 - <b>`loss`</b> (ODLoss):  Loss function. 
 - <b>`matching_function`</b>:  Function to match predictions to ground truth. 
 - <b>`alpha`</b> (float):  Risk threshold. 
 - <b>`device`</b> (str):  Device to use. 
 - <b>`B`</b> (float, optional):  Upper bound. Defaults to 1. 
 - <b>`bounds`</b> (list, optional):  Search bounds. Defaults to [0, 1]. 
 - <b>`steps`</b> (int, optional):  Number of steps for optimization. Defaults to 13. 
 - <b>`epsilon`</b> (float, optional):  Small value to ensure strict inequality. Defaults to 1e-10. 
 - <b>`verbose`</b> (bool, optional):  Whether to print progress. Defaults to False. 



**Returns:**
 
------- 
 - <b>`float`</b>:  The optimal lambda value. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If no good lambda is found. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
