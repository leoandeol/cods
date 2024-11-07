<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.models.yolo`





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `xywh2xyxy_scaled`

```python
xywh2xyxy_scaled(x, width_scale, height_scale)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AlteredYOLO`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(model_path)
```






---

#### <kbd>property</kbd> device

Retrieves the device on which the model's parameters are allocated. 

This property determines the device (CPU or GPU) where the model's parameters are currently stored. It is applicable only to models that are instances of nn.Module. 



**Returns:**
 
 - <b>`(torch.device)`</b>:  The device (CPU/GPU) of the model. 



**Raises:**
 
 - <b>`AttributeError`</b>:  If the model is not a PyTorch nn.Module instance. 



**Examples:**
 ``` model = YOLO("yolov8n.pt")```
    >>> print(model.device)
    device(type='cuda', index=0)  # if CUDA is available
    >>> model = model.to("cpu")
    >>> print(model.device)
    device(type='cpu')


---

#### <kbd>property</kbd> names

Retrieves the class names associated with the loaded model. 

This property returns the class names if they are defined in the model. It checks the class names for validity using the 'check_class_names' function from the ultralytics.nn.autobackend module. If the predictor is not initialized, it sets it up before retrieving the names. 



**Returns:**
 
 - <b>`(Dict[int, str])`</b>:  A dict of class names associated with the model. 



**Raises:**
 
 - <b>`AttributeError`</b>:  If the model or predictor does not have a 'names' attribute. 



**Examples:**
 ``` model = YOLO("yolov8n.pt")```
    >>> print(model.names)
    {0: 'person', 1: 'bicycle', 2: 'car', ...}


---

#### <kbd>property</kbd> task_map

Map head to model, trainer, validator, and predictor classes. 

---

#### <kbd>property</kbd> transforms

Retrieves the transformations applied to the input data of the loaded model. 

This property returns the transformations if they are defined in the model. The transforms typically include preprocessing steps like resizing, normalization, and data augmentation that are applied to input data before it is fed into the model. 



**Returns:**
 
 - <b>`(object | None)`</b>:  The transform object of the model if available, otherwise None. 



**Examples:**
 ``` model = YOLO("yolov8n.pt")```
    >>> transforms = model.transforms
    >>> if transforms:
    ...     print(f"Model transforms: {transforms}")
    ... else:
    ...     print("No transforms defined for this model.")




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(source=None, stream=False, **kwargs)
```






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `YOLOModel`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `postprocess`

```python
postprocess(raw_output, img_shapes, model_input_size)
```





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/yolo.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_batch`

```python
predict_batch(batch: list, **kwargs) → dict
```

Predicts the output given a batch of input tensors. 



**Args:**
 
 - <b>`batch`</b> (list):  The input batch 



**Returns:**
 
 - <b>`dict`</b>:  The predicted output as a dictionary with the following keys: 
        - "image_paths" (list): The paths of the input images 
        - "true_boxes" (list): The true bounding boxes of the objects in the images 
        - "pred_boxes" (list): The predicted bounding boxes of the objects in the images 
        - "confidences" (list): The confidence scores of the predicted bounding boxes 
        - "true_cls" (list): The true class labels of the objects in the images 
        - "pred_cls" (list): The predicted class labels of the objects in the images 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
