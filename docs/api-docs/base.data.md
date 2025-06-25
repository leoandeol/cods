<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `base.data`
Base data structures for predictions, parameters, and conformalized results. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Predictions`
Abstract base class for predictions. 

Attributes 
----------  unique_id (int): Unique ID of the predictions.  dataset_name (str): Name of the dataset.  split_name (str): Name of the split.  task_name (str): Name of the task. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    dataset_name: str,
    split_name: str,
    task_name: str,
    unique_id: Optional[int] = None
)
```

Initialize a new instance of the Predictions class. 



**Args:**
 
---- 
 - <b>`dataset_name`</b> (str):  Name of the dataset. 
 - <b>`split_name`</b> (str):  Name of the split. 
 - <b>`task_name`</b> (str):  Name of the task. 
 - <b>`unique_id`</b> (Optional[int], optional):  Unique ID of the predictions. If None, a timestamp is used. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Parameters`
Abstract base class for parameters. 

Attributes 
----------  predictions_id (int): Unique ID of the predictions.  unique_id (int): Unique ID of the parameters. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(predictions_id: int, unique_id: Optional[int] = None)
```

Initialize a new instance of the Parameters class. 



**Args:**
 
---- 
 - <b>`predictions_id`</b> (int):  Unique ID of the predictions. 
 - <b>`unique_id`</b> (Optional[int], optional):  Unique ID of the parameters. If None, a timestamp is used. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConformalizedPredictions`
Abstract base class for conformalized prediction results. 

Attributes 
----------  predictions_id (int): Unique ID of the predictions.  parameters_id (int): Unique ID of the parameters.  unique_id (int): Unique ID of the conformalized predictions. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    predictions_id: int,
    parameters_id: int,
    unique_id: Optional[int] = None
)
```

Initialize a new instance of the ConformalizedPredictions class. 



**Args:**
 
---- 
 - <b>`predictions_id`</b> (int):  Unique ID of the predictions. 
 - <b>`parameters_id`</b> (int):  Unique ID of the parameters. 
 - <b>`unique_id`</b> (Optional[int], optional):  Unique ID of the conformalized predictions. If None, a timestamp is used. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Results`
Abstract base class for results. 

Attributes 
----------  predictions_id (int): Unique ID of the predictions.  parameters_id (int): Unique ID of the parameters.  conformalized_id (int): Unique ID of the conformalized predictions. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(predictions_id: int, parameters_id: int, conformalized_id: int)
```

Initialize a new instance of the Results class. 



**Args:**
 
---- 
 - <b>`predictions_id`</b> (int):  Unique ID of the predictions. 
 - <b>`parameters_id`</b> (int):  Unique ID of the parameters. 
 - <b>`conformalized_id`</b> (int):  Unique ID of the conformalized predictions. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
