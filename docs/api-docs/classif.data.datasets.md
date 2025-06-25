<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/data/datasets.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `classif.data.datasets`
Datasets for conformal classification tasks. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/data/datasets.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassificationDataset`
Dataset for classification tasks with class index mapping and optional transforms. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/data/datasets.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    path: str,
    transforms: Callable = None,
    idx_to_cls: Dict[int, str] = None,
    **kwargs
)
```

Initialize the ClassificationDataset. 



**Args:**
 
---- 
 - <b>`path`</b> (str):  Path to the dataset. 
 - <b>`transforms`</b> (Callable, optional):  Transformations to apply to images. 
 - <b>`idx_to_cls`</b> (dict, optional):  Mapping from class indices to class names. 
 - <b>`**kwargs`</b>:  Additional arguments for the base dataset. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If idx_to_cls is not provided. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/data/datasets.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `random_split`

```python
random_split(lengths, seed=0)
```

Randomly split the dataset into subsets of given lengths. 



**Args:**
 
---- 
 - <b>`lengths`</b> (list):  Lengths of splits. 
 - <b>`seed`</b> (int, optional):  Random seed. Defaults to 0. 



**Yields:**
 
------ 
 - <b>`ClassificationDataset`</b>:  Subsets of the dataset. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If idx_to_cls is not set in the split. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/data/datasets.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ImageNetDataset`
Dataset for ImageNet with automatic class index mapping and default transforms. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/data/datasets.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(path: str, transforms: Callable = None, **kwargs)
```

Initialize the ImageNetDataset. 



**Args:**
 
---- 
 - <b>`path`</b> (str):  Path to the dataset. 
 - <b>`transforms`</b> (Callable, optional):  Transformations to apply to images. 
 - <b>`**kwargs`</b>:  Additional arguments for the base dataset. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/data/datasets.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `random_split`

```python
random_split(lengths, seed=0)
```

Randomly split the dataset into subsets of given lengths. 



**Args:**
 
---- 
 - <b>`lengths`</b> (list):  Lengths of splits. 
 - <b>`seed`</b> (int, optional):  Random seed. Defaults to 0. 



**Yields:**
 
------ 
 - <b>`ClassificationDataset`</b>:  Subsets of the dataset. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If idx_to_cls is not set in the split. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
