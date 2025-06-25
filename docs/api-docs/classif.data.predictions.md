<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/data/predictions.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `classif.data.predictions`






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/data/predictions.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassificationPredictions`
Container for predictions from a classification model. 

Stores image paths, true and predicted class labels, and class index mapping for a classification task. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/data/predictions.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    dataset_name: str,
    split_name: str,
    image_paths: List[str],
    idx_to_cls: Optional[Dict[int, str]],
    true_cls: Tensor,
    pred_cls: Tensor
)
```

Initialize ClassificationPredictions. 



**Args:**
 
---- 
 - <b>`dataset_name`</b> (str):  Name of the dataset. 
 - <b>`split_name`</b> (str):  Name of the data split (e.g., 'train', 'val'). 
 - <b>`image_paths`</b> (List[str]):  List of image file paths. 
 - <b>`idx_to_cls`</b> (dict or None):  Mapping from class indices to class names. 
 - <b>`true_cls`</b> (torch.Tensor):  Ground truth class labels (N,). 
 - <b>`pred_cls`</b> (torch.Tensor):  Model predictions (N, num_classes), before softmax. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/data/predictions.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `split`

```python
split(splits_names: list, splits_ratios: list)
```

Split predictions into multiple splits. 



**Args:**
 
---- 
 - <b>`splits_names`</b> (list):  List of names for each split. 
 - <b>`splits_ratios`</b> (list):  List of ratios for each split. Must sum to 1. 



**Returns:**
 
------- 
 - <b>`list`</b>:  List of ClassificationPredictions objects, one for each split. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
