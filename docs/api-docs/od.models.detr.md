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

Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format. 



**Args:**
 
 - <b>`x`</b> (torch.Tensor):  Bounding boxes in (cx, cy, w, h) format. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Bounding boxes in (x1, y1, x2, y2) format. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/detr.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `box_xyxy_to_cxcywh`

```python
box_xyxy_to_cxcywh(x)
```

Convert bounding boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format. 



**Args:**
 
 - <b>`x`</b> (torch.Tensor):  Bounding boxes in (x1, y1, x2, y2) format. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Bounding boxes in (cx, cy, w, h) format. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/detr.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rescale_bboxes`

```python
rescale_bboxes(out_bbox, size)
```

Rescale bounding boxes to image size. 



**Args:**
 
 - <b>`out_bbox`</b> (torch.Tensor):  Normalized bounding boxes. 
 - <b>`size`</b> (tuple):  Image size (width, height). 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  Rescaled bounding boxes in absolute coordinates. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/detr.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DETRModel`
DETR (DEtection TRansformer) model wrapper for object detection. 

Provides a wrapper around the DETR model with preprocessing, postprocessing, and prediction generation capabilities for conformal prediction workflows. 



**Attributes:**
 
 - <b>`MODEL_NAMES`</b> (list):  List of supported DETR model variants. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/detr.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Initialize the DETR model. 



**Args:**
 
 - <b>`model_name`</b> (str, optional):  Name of the DETR model variant. Defaults to 'detr_resnet50'. 
 - <b>`pretrained`</b> (bool, optional):  Whether to use pretrained weights. Defaults to True. 
 - <b>`weights`</b> (str, optional):  Path to custom weights. Defaults to None. 
 - <b>`device`</b> (str, optional):  Device to run the model on. Defaults to 'cpu'. 
 - <b>`save`</b> (bool, optional):  Whether to save predictions. Defaults to True. 
 - <b>`save_dir_path`</b> (str, optional):  Directory to save predictions. Defaults to None. 



**Raises:**
 
 - <b>`ValueError`</b>:  If model_name is not in MODEL_NAMES. 
 - <b>`NotImplementedError`</b>:  If pretrained is False (only pretrained models supported). 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/detr.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `postprocess`

```python
postprocess(outputs, image_sizes)
```

Post-process model outputs to extract predictions. 



**Args:**
 
 - <b>`outputs`</b> (dict):  Raw model outputs containing 'pred_logits' and 'pred_boxes'. 
 - <b>`image_sizes`</b> (torch.Tensor):  Image sizes for rescaling bounding boxes. 



**Returns:**
 
 - <b>`tuple`</b>:  (scaled_pred_boxes, confidences, pred_cls) where: 
        - scaled_pred_boxes: Rescaled bounding boxes 
        - confidences: Confidence scores 
        - pred_cls: Class probabilities 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/detr.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_batch`

```python
predict_batch(batch: list, **kwargs) → dict
```

Predicts the output given a batch of input tensors. 



**Args:**
 
---- 
 - <b>`batch`</b> (list):  The input batch 
 - <b>`**kwargs`</b>:  Additional keyword arguments passed to the prediction method 



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
