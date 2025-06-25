<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.models.utils`
Utility functions and classes for object detection models. 

This module provides utility functions and classes for object detection models, including channel resizing, Bayesian object detection postprocessing, and prediction filtering utilities. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/utils.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bayesod`

```python
bayesod(
    pred_boxes: Tensor,
    confidences: Tensor,
    pred_cls: Tensor,
    iou_threshold: float
)
```

_summary_. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  _description_ 
 - <b>`confidences`</b> (torch.Tensor):  _description_ 
 - <b>`pred_cls`</b> (torch.Tensor):  _description_ 
 - <b>`iou_threshold`</b> (float):  _description_ 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/utils.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `filter_preds`

```python
filter_preds(preds, confidence_threshold=0.001)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/utils.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ResizeChannels`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/utils.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(num_channels)
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/utils.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(image)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
