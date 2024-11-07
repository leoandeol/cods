<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.loss`






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(upper_bound: int, **kwargs)
```

Initialize the Object Detection Loss. 

Parameters 
---------- 
- upper_bound (int): The upper bound of the loss. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxCountThresholdConfidenceLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    upper_bound: int = 1,
    other_losses: Optional[List[Loss]] = None,
    **kwargs
)
```

Initialize the Confidence Loss. 

Parameters 
---------- 
- upper_bound (int): The upper bound of the loss. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxCountRecallConfidenceLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    upper_bound: int = 1,
    distance_threshold: float = 100,
    other_losses: Optional[List[Loss]] = None,
    **kwargs
)
```

Initialize the Confidence Loss. 

Parameters 
---------- 
- upper_bound (int): The upper bound of the loss. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L214"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODBinaryClassificationLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```









---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassificationLossWrapper`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L237"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(classification_loss, **kwargs)
```

Initialize the Classification Loss Wrapper. 

Parameters 
---------- 
- classification_loss (Loss): The classification loss. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L313"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ThresholdedRecallLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(beta: float = 0.25)
```

Initialize the Hausdorff Signed Distance Loss. 

Parameters 
---------- 
- beta (float): The beta value. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L366"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassBoxWiseRecallLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L367"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: bool = True)
```

Initialize the Box-wise Recall Loss. 

Parameters 
---------- 
- union_of_boxes (bool): Whether to use the union of boxes. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L433"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxWiseRecallLoss`
Box-wise recall loss: 1 - mean(areas of the union of the boxes), 

This loss function calculates the recall loss based on the areas of the union of the predicted and true bounding boxes. The recall loss is defined as 1 minus the mean of the areas of the union of the boxes. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L440"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: bool = True)
```

Initialize the Box-wise Recall Loss. 

Parameters 
---------- 
- union_of_boxes (bool): Whether to use the union of boxes. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L494"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PixelWiseRecallLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L495"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: bool = True)
```

Initialize the Pixel-wise Recall Loss. 

Parameters 
---------- 
- union_of_boxes (bool): Whether to use the union of boxes. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L544"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NumberPredictionsGapLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L545"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the Number Predictions Gap Loss. 

Returns 
------- 
- None 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
