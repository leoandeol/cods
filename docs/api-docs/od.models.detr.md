<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/detr.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.models.detr`
DETR (DEtection TRansformer) model implementation for object detection. 

This module provides the DETR model wrapper for object detection with conformal prediction support, including model loading, prediction generation, and post-processing utilities for bounding box transformations. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/detr.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `box_cxcywh_to_xyxy`

```python
box_cxcywh_to_xyxy(x)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/detr.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `box_xyxy_to_cxcywh`

```python
box_xyxy_to_cxcywh(x)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/detr.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rescale_bboxes`

```python
rescale_bboxes(out_bbox, size)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/detr.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DETRModel`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/detr.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model_name='detr_resnet50',
    pretrained=True,
    weights=None,
    device='cpu',
    save=True,
    save_dir_path=None
)
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/detr.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `postprocess`

```python
postprocess(outputs, image_sizes)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/detr.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_batch`

```python
predict_batch(batch: list, **kwargs) → dict
```

Predicts the output given a batch of input tensors. 



**Args:**
 
---- 
 - <b>`batch`</b> (list):  The input batch 



**Returns:**
 
------- 
 - <b>`dict`</b>:  The predicted output as a dictionary with the following keys: 
        - "image_paths" (list): The paths of the input images 
        - "true_boxes" (list): The true bounding boxes of the objects in the images 
        - "pred_boxes" (list): The predicted bounding boxes of the objects in the images 
        - "confidences" (list): The confidence scores of the predicted bounding boxes 
        - "true_cls" (list): The true class labels of the objects in the images 
        - "pred_cls" (list): The predicted class labels of the objects in the images 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
