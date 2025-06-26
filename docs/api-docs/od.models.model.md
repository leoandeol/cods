<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.models.model`
Base object detection model class for conformal prediction. 

This module provides the abstract base class for object detection models, defining the interface for prediction generation, model loading, and integration with the conformal prediction framework. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/model.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODModel`
Abstract base class for object detection models. 

Provides the interface and common functionality for object detection models used in conformal prediction workflows, including prediction building, caching, and post-processing capabilities. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/model.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model_name: str,
    save_dir_path: str,
    pretrained: bool = True,
    weights: Optional[str] = None,
    device: str = 'cpu'
)
```

Initialize an instance of the ODModel class. 



**Args:**
 
---- 
 - <b>`model_name`</b> (str):  The name of the model. 
 - <b>`save_dir_path`</b> (str):  The path to save the model. 
 - <b>`pretrained`</b> (bool, optional):  Whether to use pretrained weights. Defaults to True. 
 - <b>`weights`</b> (str, optional):  The path to the weights file. Defaults to None. 
 - <b>`device`</b> (str, optional):  The device to use for computation. Defaults to "cpu". 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/model.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `build_predictions`

```python
build_predictions(
    dataset,
    dataset_name: str,
    split_name: str,
    batch_size: int,
    shuffle: bool = False,
    verbose: bool = True,
    force_recompute: bool = False,
    deletion_method: str = 'nms',
    iou_threshold: float = 0.5,
    filter_preds_by_confidence: Optional[float] = None,
    **kwargs
) → ODPredictions
```

Build predictions for the given dataset. 



**Args:**
 
---- 
 - <b>`dataset`</b>:  The dataset to build predictions for. 
 - <b>`dataset_name`</b> (str):  The name of the dataset. 
 - <b>`split_name`</b> (str):  The name of the split. 
 - <b>`batch_size`</b> (int):  The batch size for prediction. 
 - <b>`shuffle`</b> (bool, optional):  Whether to shuffle the dataset. Defaults to False. 
 - <b>`verbose`</b> (bool, optional):  Prints progress. Defaults to True. 
 - <b>`force_recompute`</b> (bool, optional):  Whether to force recomputation of predictions. Defaults to False. 
 - <b>`deletion_method`</b> (str, optional):  Method for deleting redundant boxes. Defaults to "nms". 
 - <b>`iou_threshold`</b> (float, optional):  IoU threshold for filtering. Defaults to 0.5. 
 - <b>`filter_preds_by_confidence`</b> (float, optional):  Confidence threshold for filtering predictions. Defaults to None. 
 - <b>`**kwargs`</b>:  Additional keyword arguments for the DataLoader. 



**Returns:**
 
------- 
 - <b>`ODPredictions`</b>:  Predictions object to use for prediction set construction. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/models/model.py#L293"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_batch`

```python
predict_batch(batch: list, **kwargs) → dict
```

Predicts the output given a batch of input tensors. 



**Args:**
 
---- 
 - <b>`batch`</b> (list):  The input batch. 
 - <b>`**kwargs`</b>:  Additional keyword arguments passed to the prediction method. 



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
