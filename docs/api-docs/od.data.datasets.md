<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/datasets.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.data.datasets`
Dataset classes for object detection tasks. 

This module provides dataset implementations for object detection, including MS-COCO and VOC datasets with support for conformal prediction workflows. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/datasets.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MSCOCODataset`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/datasets.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(root, split, transforms=None, image_ids=None)
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/datasets.py#L242"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `shuffle`

```python
shuffle()
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/datasets.py#L245"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `split_dataset`

```python
split_dataset(proportion, shuffle=False, n_calib_test: Optional[int] = None)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/datasets.py#L283"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VOCDataset`





---

#### <kbd>property</kbd> annotations










---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
