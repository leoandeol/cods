<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/datasets.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.data.datasets`
Dataset classes for object detection tasks. 

This module provides dataset implementations for object detection, including MS-COCO and VOC datasets with support for conformal prediction workflows. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/datasets.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MSCOCODataset`
MS-COCO dataset implementation for object detection. 

Provides access to the MS-COCO dataset with support for train/val/test splits, image loading, annotation handling, and dataset splitting for conformal prediction. 



**Attributes:**
 
 - <b>`NAMES`</b> (dict):  Mapping from class IDs to class names for MS-COCO dataset. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/datasets.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(root, split, transforms=None, image_ids=None)
```

Initialize the MS-COCO dataset. 



**Args:**
 
 - <b>`root`</b> (str):  Root directory path to the MS-COCO dataset. 
 - <b>`split`</b> (str):  Dataset split ('train', 'val', or 'test'). 
 - <b>`transforms`</b> (callable, optional):  Transformations to apply to images. 
 - <b>`image_ids`</b> (list, optional):  Specific image IDs to use. If None, uses all images. 



**Raises:**
 
 - <b>`ValueError`</b>:  If split is not one of ['train', 'val', 'test']. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/datasets.py#L292"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `shuffle`

```python
shuffle()
```

Randomly shuffle the order of images in the dataset. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/datasets.py#L296"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `split_dataset`

```python
split_dataset(proportion, shuffle=False, n_calib_test: Optional[int] = None)
```

Split the dataset into two parts. 



**Args:**
 
 - <b>`proportion`</b> (float):  Proportion of data for the first split (0-1). 
 - <b>`shuffle`</b> (bool, optional):  Whether to shuffle before splitting. Defaults to False. 
 - <b>`n_calib_test`</b> (int, optional):  Maximum number of samples to use for splitting.  If None, uses all samples. 



**Returns:**
 
 - <b>`tuple`</b>:  (dataset1, dataset2) where dataset1 contains the first `proportion`  of the data and dataset2 contains the remaining data. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/data/datasets.py#L347"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VOCDataset`
PASCAL VOC dataset implementation for object detection. 

Extends torchvision's VOCDetection to provide consistent interface with other dataset classes in the CODS framework. 



**Attributes:**
 
 - <b>`VOC_CLASSES`</b> (list):  List of class names in PASCAL VOC dataset. 


---

#### <kbd>property</kbd> annotations










---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
