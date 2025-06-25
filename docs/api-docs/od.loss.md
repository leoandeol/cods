<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.loss`
Loss functions for object detection conformal prediction. 

This module implements various loss functions used in object detection conformal prediction, including localization losses, size-based losses, and classification losses adapted for object detection tasks. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODLoss`
Base class for Object Detection losses. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(upper_bound: 'int', device: 'str' = 'cpu', **kwargs)
```

Initialize the Object Detection Loss. 



**Args:**
 
---- 
 - <b>`upper_bound`</b> (int):  The upper bound of the loss. 
 - <b>`device`</b> (str, optional):  Device to use for tensors. Defaults to "cpu". 
 - <b>`**kwargs`</b>:  Additional keyword arguments. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxCountThresholdConfidenceLoss`
Confidence loss based on whether the count of conformalized boxes meets or exceeds the count of true boxes. 

The loss is 0 if `len(conf_boxes) >= len(true_boxes)`, and 1 otherwise. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(upper_bound: 'int' = 1, device: 'str' = 'cpu', **kwargs)
```

Initialize the BoxCountThresholdConfidenceLoss. 



**Args:**
 
---- 
 - <b>`upper_bound`</b> (int, optional):  The upper bound of the loss. Defaults to 1. 
 - <b>`device`</b> (str, optional):  Device to use for tensors. Defaults to "cpu". 
 - <b>`**kwargs`</b>:  Additional keyword arguments. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxCountTwosidedConfidenceLoss`
Confidence loss based on whether the absolute difference between true and predicted box counts exceeds a threshold. 

The loss is 1 if `abs(len(true_boxes) - len(conf_boxes)) > self.threshold`, and 0 otherwise. If there are no true boxes, the loss is 0. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    upper_bound: 'int' = 1,
    threshold: 'int' = 3,
    device: 'str' = 'cpu',
    **kwargs
)
```

Initialize the BoxCountTwosidedConfidenceLoss. 



**Args:**
 
---- 
 - <b>`upper_bound`</b> (int, optional):  The upper bound of the loss. Defaults to 1. 
 - <b>`threshold`</b> (int, optional):  Allowed difference in box counts. Defaults to 3. 
 - <b>`device`</b> (str, optional):  Device to use for tensors. Defaults to "cpu". 
 - <b>`**kwargs`</b>:  Additional keyword arguments. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxCountRecallConfidenceLoss`
Confidence loss based on the recall of box counts. 

Calculates `max(0, (len(true_boxes) - len(conf_boxes)) / len(true_boxes))`. If there are no true boxes, the loss is 0. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(upper_bound: 'int' = 1, device: 'str' = 'cpu', **kwargs)
```

Initialize the BoxCountRecallConfidenceLoss. 



**Args:**
 
---- 
 - <b>`upper_bound`</b> (int, optional):  The upper bound of the loss. Defaults to 1. 
 - <b>`device`</b> (str, optional):  Device to use for tensors. Defaults to "cpu". 
 - <b>`**kwargs`</b>:  Additional keyword arguments. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L238"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ThresholdedBoxDistanceConfidenceLoss`
Confidence loss based on a thresholded distance between true and predicted boxes. 

This loss computes a combined distance (Hausdorff and LAC) between true and predicted boxes. The loss is the mean of indicators where this distance exceeds `self.distance_threshold`. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L245"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    upper_bound: 'int' = 1,
    distance_threshold: 'float' = 0.5,
    device: 'str' = 'cpu',
    **kwargs
)
```

Initialize the ThresholdedBoxDistanceConfidenceLoss. 



**Args:**
 
---- 
 - <b>`upper_bound`</b> (int, optional):  The upper bound of the loss. Defaults to 1. 
 - <b>`distance_threshold`</b> (float, optional):  Distance threshold for loss. Defaults to 0.5. 
 - <b>`device`</b> (str, optional):  Device to use for tensors. Defaults to "cpu". 
 - <b>`**kwargs`</b>:  Additional keyword arguments. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L308"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODBinaryClassificationLoss`
Binary classification loss for object detection. 

This loss is 1 if the true class is not in the conformalized class set, and 0 otherwise. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the ODBinaryClassificationLoss. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L343"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassificationLossWrapper`
Wraps a standard classification loss for use in object detection. 

This class applies a given classification loss to each true object and its corresponding conformalized class predictions, then averages the losses. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L350"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(classification_loss, device: 'str' = 'cpu', **kwargs)
```

Initialize the ClassificationLossWrapper. 



**Args:**
 
---- 
 - <b>`classification_loss`</b> (Loss):  The classification loss to wrap. 
 - <b>`device`</b> (str, optional):  Device to use for tensors. Defaults to "cpu". 
 - <b>`**kwargs`</b>:  Additional keyword arguments. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L405"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ThresholdedRecallLoss`
A recall loss that is 1 if the miscoverage (1 - recall) exceeds a threshold `beta`, and 0 otherwise. 

Miscoverage is calculated based on the proportion of true boxes not sufficiently covered by the union of conformalized boxes. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L412"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(beta: 'float' = 0.25, device: 'str' = 'cpu')
```

Initialize the ThresholdedRecallLoss. 



**Args:**
 
---- 
 - <b>`beta`</b> (float, optional):  The beta value. Defaults to 0.25. 
 - <b>`device`</b> (str, optional):  Device to use for tensors. Defaults to "cpu". 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L466"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassBoxWiseRecallLoss`
A combined recall loss for both localization (box-wise recall) and classification. 

The loss is the mean of indicators where either the true box is not sufficiently covered by the union of conformalized boxes, OR the true class is not in the conformalized class set. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L473"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: 'bool' = True, device: 'str' = 'cpu')
```

Initialize the ClassBoxWiseRecallLoss. 



**Args:**
 
---- 
 - <b>`union_of_boxes`</b> (bool, optional):  Whether to use the union of boxes. Defaults to True. 
 - <b>`device`</b> (str, optional):  Device to use for tensors. Defaults to "cpu". 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L541"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxWiseRecallLoss`
Box-wise recall loss: 1 - mean(areas of the union of the boxes). 

This loss function calculates the recall loss based on the areas of the union of the predicted and true bounding boxes. The recall loss is defined as 1 minus the mean of the areas of the union of the boxes. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L548"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: 'bool' = True, device: 'str' = 'cpu')
```

Initialize the BoxWiseRecallLoss. 



**Args:**
 
---- 
 - <b>`union_of_boxes`</b> (bool, optional):  Whether to use the union of boxes. Defaults to True. 
 - <b>`device`</b> (str, optional):  Device to use for tensors. Defaults to "cpu". 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L612"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PixelWiseRecallLoss`
Pixel-wise recall loss. 

Calculates `1 - mean(areas)`, where `areas` are the fractions of each true box covered by the corresponding (matched) conformalized box. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L619"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: 'bool' = True, device: 'str' = 'cpu')
```

Initialize the PixelWiseRecallLoss. 



**Args:**
 
---- 
 - <b>`union_of_boxes`</b> (bool, optional):  Whether to use the union of boxes. Defaults to True. 
 - <b>`device`</b> (str, optional):  Device to use for tensors. Defaults to "cpu". 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L673"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxWisePrecisionLoss`
Box-wise precision loss. 

For each conformalized box, it finds the maximum overlap (area of contained part of true box) with any true box. The loss is the mean of indicators where this maximum overlap is insufficient (e.g., < 0.999). 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L680"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: 'bool' = True, device: 'str' = 'cpu')
```

Initialize the BoxWisePrecisionLoss. 



**Args:**
 
---- 
 - <b>`union_of_boxes`</b> (bool, optional):  Whether to use the union of boxes. Defaults to True. 
 - <b>`device`</b> (str, optional):  Device to use for tensors. Defaults to "cpu". 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L740"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxWiseIoULoss`
Box-wise IoU loss. 

Calculates the mean of indicators where the Generalized IoU (GIoU) between true boxes and conformalized boxes is less than a threshold (e.g., 0.9). 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L747"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: 'bool' = True, device: 'str' = 'cpu')
```

Initialize the BoxWiseIoULoss. 



**Args:**
 
---- 
 - <b>`union_of_boxes`</b> (bool, optional):  Whether to use the union of boxes. Defaults to True. 
 - <b>`device`</b> (str, optional):  Device to use for tensors. Defaults to "cpu". 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L806"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NumberPredictionsGapLoss`
Loss based on the normalized difference between the number of true boxes and conformalized boxes. 

Calculates `(len(true_boxes) - len(conf_boxes)) / max(len(true_boxes), 1)`, capped at 1. Note: This loss is currently not implemented. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L813"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(device: 'str' = 'cpu')
```

Initialize the NumberPredictionsGapLoss. 



**Args:**
 
---- 
 - <b>`device`</b> (str, optional):  Device to use for tensors. Defaults to "cpu". 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
