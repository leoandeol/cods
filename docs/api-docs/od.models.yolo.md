<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.models.yolo`
YOLO model implementation for object detection with conformal prediction. 

This module provides the YOLO model wrapper for object detection tasks, including model loading, prediction generation, and post-processing utilities for bounding box format conversion and scaling. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `xywh2xyxy_scaled`

```python
xywh2xyxy_scaled(x, width_scale, height_scale)
```

Convert bounding boxes from center (x, y, w, h) format to (x0, y0, x1, y1) format and scale. 



**Args:**
 
---- 
 - <b>`x`</b> (torch.Tensor):  Bounding boxes in (center_x, center_y, width, height) format. 
 - <b>`width_scale`</b> (float):  Scaling factor for width. 
 - <b>`height_scale`</b> (float):  Scaling factor for height. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Bounding boxes in (x0, y0, x1, y1) format, scaled. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlteredYOLO`
YOLO model wrapper with hooks to capture raw outputs and input shapes during prediction. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(model_path)
```

Initialize the AlteredYOLO model. 



**Args:**
 
---- 
 - <b>`model_path`</b> (str):  Path to the YOLO model weights. 


---

#### <kbd>property</kbd> device

Get the device on which the model's parameters are allocated. 

This property determines the device (CPU or GPU) where the model's parameters are currently stored. It is applicable only to models that are instances of torch.nn.Module. 



**Returns:**
 
 - <b>`(torch.device)`</b>:  The device (CPU/GPU) of the model. 



**Raises:**
 
 - <b>`AttributeError`</b>:  If the model is not a torch.nn.Module instance. 



**Examples:**
 ``` model = YOLO("yolo11n.pt")```
    >>> print(model.device)
    device(type='cuda', index=0)  # if CUDA is available
    >>> model = model.to("cpu")
    >>> print(model.device)
    device(type='cpu')


---

#### <kbd>property</kbd> names

Retrieve the class names associated with the loaded model. 

This property returns the class names if they are defined in the model. It checks the class names for validity using the 'check_class_names' function from the ultralytics.nn.autobackend module. If the predictor is not initialized, it sets it up before retrieving the names. 



**Returns:**
 
 - <b>`(Dict[int, str])`</b>:  A dictionary of class names associated with the model, where keys are class indices and  values are the corresponding class names. 



**Raises:**
 
 - <b>`AttributeError`</b>:  If the model or predictor does not have a 'names' attribute. 



**Examples:**
 ``` model = YOLO("yolo11n.pt")```
    >>> print(model.names)
    {0: 'person', 1: 'bicycle', 2: 'car', ...}


---

#### <kbd>property</kbd> task_map

Map head to model, trainer, validator, and predictor classes. 

---

#### <kbd>property</kbd> transforms

Retrieve the transformations applied to the input data of the loaded model. 

This property returns the transformations if they are defined in the model. The transforms typically include preprocessing steps like resizing, normalization, and data augmentation that are applied to input data before it is fed into the model. 



**Returns:**
 
 - <b>`(object | None)`</b>:  The transform object of the model if available, otherwise None. 



**Examples:**
 ``` model = YOLO("yolo11n.pt")```
    >>> transforms = model.transforms
    >>> if transforms:
    ...     print(f"Model transforms: {transforms}")
    ... else:
    ...     print("No transforms defined for this model.")




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(source=None, stream=False, **kwargs)
```

Run prediction and capture raw outputs and input shapes using hooks. 



**Args:**
 
---- 
 - <b>`source`</b>:  Input source for prediction. 
 - <b>`stream`</b> (bool, optional):  Whether to stream results. Defaults to False. 
 - <b>`**kwargs`</b>:  Additional keyword arguments for prediction. 



**Returns:**
 
------- The results from YOLO prediction. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `YOLOModel`
Object Detection model wrapper for YOLO with custom preprocessing and postprocessing. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model_name='yolov8x.pt',
    pretrained=True,
    weights=None,
    device='cpu',
    save=True,
    save_dir_path=None
)
```

Initialize the YOLOModel. 



**Args:**
 
---- 
 - <b>`model_name`</b> (str, optional):  Name or path of the YOLO model. Defaults to "yolov8x.pt". 
 - <b>`pretrained`</b> (bool, optional):  Whether to use pretrained weights. Defaults to True. 
 - <b>`weights`</b>:  Custom weights (not used currently). 
 - <b>`device`</b> (str, optional):  Device to use. Defaults to "cpu". 
 - <b>`save`</b> (bool, optional):  Whether to save the model. Defaults to True. 
 - <b>`save_dir_path`</b> (str, optional):  Directory to save the model. Defaults to None. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `postprocess`

```python
postprocess(raw_output, img_shapes, model_input_size)
```

Postprocess raw model outputs to obtain bounding boxes, confidences, and class probabilities. 



**Args:**
 
---- 
 - <b>`raw_output`</b> (list[torch.Tensor]):  Raw outputs from the model. 
 - <b>`img_shapes`</b> (torch.FloatTensor):  Original image shapes. 
 - <b>`model_input_size`</b> (tuple):  Model input size (width, height). 



**Returns:**
 
------- 
 - <b>`tuple`</b>:  (all_boxes, all_confs, all_probs) for each image in the batch. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L294"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_batch`

```python
predict_batch(batch: list, **kwargs) → dict
```

Predict the output given a batch of input tensors. 



**Args:**
 
---- 
 - <b>`batch`</b> (list):  The input batch containing image paths, image sizes, images, and ground truth. 
 - <b>`**kwargs`</b>:  Additional keyword arguments for prediction. 



**Returns:**
 
------- 
 - <b>`dict`</b>:  The predicted output as a dictionary with the following keys: 
        - "image_paths" (list): The paths of the input images 
        - "image_shapes" (list): The shapes of the input images 
        - "true_boxes" (list): The true bounding boxes of the objects in the images 
        - "pred_boxes" (list): The predicted bounding boxes of the objects in the images 
        - "confidences" (list): The confidence scores of the predicted bounding boxes 
        - "true_cls" (list): The true class labels of the objects in the images 
        - "pred_cls" (list): The predicted class labels of the objects in the images 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
