<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/models.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `base.models`
Base model class for machine learning models in the cods library. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/models.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Model`
Abstract base class for models in the cods library. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/models.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model_name: str,
    save_dir_path: str,
    pretrained=True,
    weights=None,
    device='cpu'
)
```

Initialize the Model base class. 



**Args:**
 
---- 
 - <b>`model_name`</b> (str):  Name of the model. 
 - <b>`save_dir_path`</b> (str):  Directory path to save predictions. 
 - <b>`pretrained`</b> (bool, optional):  Whether to use pretrained weights. Defaults to True. 
 - <b>`weights`</b> (optional):  Model weights. Defaults to None. 
 - <b>`device`</b> (str, optional):  Device to use. Defaults to 'cpu'. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/base/models.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `build_predictions`

```python
build_predictions(dataloader: DataLoader, verbose=True, **kwargs) → Predictions
```

Build predictions for the given dataloader. 



**Args:**
 
---- 
 - <b>`dataloader`</b> (torch.utils.data.DataLoader):  DataLoader to generate predictions from. 
 - <b>`verbose`</b> (bool, optional):  Whether to print progress. Defaults to True. 
 - <b>`**kwargs`</b>:  Additional arguments. 



**Returns:**
 
------- 
 - <b>`Predictions`</b>:  Predictions object. 



**Raises:**
 
------ 
 - <b>`NotImplementedError`</b>:  If not implemented in subclass. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
