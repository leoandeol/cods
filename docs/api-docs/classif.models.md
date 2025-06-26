<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/models.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `classif.models`
Model wrapper for conformal classification tasks. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/models.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassificationModel`
Model wrapper for classification tasks with prediction saving/loading. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/models.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model,
    model_name,
    pretrained=True,
    weights=None,
    device='cpu',
    save=True,
    save_dir_path=None
)
```

Initialize the ClassificationModel. 



**Args:**
 
---- 
 - <b>`model`</b>:  The underlying PyTorch model. 
 - <b>`model_name`</b> (str):  Name of the model. 
 - <b>`pretrained`</b> (bool, optional):  Whether to use pretrained weights. Defaults to True. 
 - <b>`weights`</b> (optional):  Model weights. Defaults to None. 
 - <b>`device`</b> (str, optional):  Device to use. Defaults to 'cpu'. 
 - <b>`save`</b> (bool, optional):  Whether to save predictions. Defaults to True. 
 - <b>`save_dir_path`</b> (str, optional):  Directory to save predictions. Defaults to None. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/models.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `build_predictions`

```python
build_predictions(
    dataset,
    dataset_name: str,
    split_name: str,
    batch_size: int,
    shuffle: bool = False,
    verbose: bool = True,
    **kwargs
) → ClassificationPredictions
```

Build predictions for the given dataset and save/load as needed. 



**Args:**
 
---- 
 - <b>`dataset`</b>:  Dataset to build predictions for. 
 - <b>`dataset_name`</b> (str):  Name of the dataset. 
 - <b>`split_name`</b> (str):  Name of the data split. 
 - <b>`batch_size`</b> (int):  Batch size for prediction. 
 - <b>`shuffle`</b> (bool, optional):  Whether to shuffle the data. Defaults to False. 
 - <b>`verbose`</b> (bool, optional):  Whether to print progress. Defaults to True. 
 - <b>`**kwargs`</b>:  Additional arguments for DataLoader. 



**Returns:**
 
------- 
 - <b>`ClassificationPredictions`</b>:  Predictions object for the dataset. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
