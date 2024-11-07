<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `base.data`






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Predictions`
Abstract class for predictions 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    dataset_name: str,
    split_name: str,
    task_name: str,
    unique_id: Optional[int] = None
)
```

Initializes a new instance of the Predictions class. 

**Args:**
 
 - <b>`unique_id`</b> (int):  The unique ID of the predictions. 
 - <b>`dataset_name`</b> (str):  The name of the dataset. 
 - <b>`split_name`</b> (str):  The name of the split. 
 - <b>`task_name`</b> (str):  The name of the task. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Parameters`
Abstract class for parameters 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(predictions_id: int, unique_id: Optional[int] = None)
```

Initializes a new instance of the Parameters class. 



**Parameters:**
 predictions_id (int): The unique ID of the predictions. unique_id (int): The unique ID of the parameters. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConformalizedPredictions`
Abstract class for results 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    predictions_id: int,
    parameters_id: int,
    unique_id: Optional[int] = None
)
```

Initializes a new instance of the Data class. 



**Parameters:**
 predictions_id (int): The unique ID of the predictions. parameters_id (int): The unique ID of the parameters. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Results`
Abstract class for results 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/data.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(predictions_id: int, parameters_id: int, conformalized_id: int)
```

Initializes a new instance of the Data class. 



**Parameters:**
 predictions_id (int): The unique ID of the predictions. parameters_id (int): The unique ID of the parameters. conformalized_id (int): The unique ID of the conformalized predictions. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
