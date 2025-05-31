<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.loss`






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(upper_bound: 'int', device: 'str' = 'cpu', **kwargs)
```

Initialize the Object Detection Loss. 

Parameters 
---------- 
- upper_bound (int): The upper bound of the loss. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxCountThresholdConfidenceLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(upper_bound: 'int' = 1, device: 'str' = 'cpu', **kwargs)
```

Initialize the Confidence Loss. 

Parameters 
---------- 
- upper_bound (int): The upper bound of the loss. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxCountTwosidedConfidenceLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    upper_bound: 'int' = 1,
    threshold: 'int' = 3,
    device: 'str' = 'cpu',
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

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxCountRecallConfidenceLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(upper_bound: 'int' = 1, device: 'str' = 'cpu', **kwargs)
```

Initialize the Confidence Loss. 

Parameters 
---------- 
- upper_bound (int): The upper bound of the loss. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L261"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ThresholdedBoxDistanceConfidenceLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L262"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    upper_bound: 'int' = 1,
    distance_threshold: 'float' = 0.5,
    device: 'str' = 'cpu',
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

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L324"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODBinaryClassificationLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L325"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```









---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L346"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassificationLossWrapper`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L347"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(classification_loss, device: 'str' = 'cpu', **kwargs)
```

Initialize the Classification Loss Wrapper. 

Parameters 
---------- 
- classification_loss (Loss): The classification loss. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L428"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ThresholdedRecallLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L429"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(beta: 'float' = 0.25, device: 'str' = 'cpu')
```

Initialize the Hausdorff Signed Distance Loss. 

Parameters 
---------- 
- beta (float): The beta value. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L484"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassBoxWiseRecallLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L485"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: 'bool' = True, device: 'str' = 'cpu')
```

Initialize the Box-wise Recall Loss. 

Parameters 
---------- 
- union_of_boxes (bool): Whether to use the union of boxes. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L557"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxWiseRecallLoss`
Box-wise recall loss: 1 - mean(areas of the union of the boxes), 

This loss function calculates the recall loss based on the areas of the union of the predicted and true bounding boxes. The recall loss is defined as 1 minus the mean of the areas of the union of the boxes. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L564"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: 'bool' = True, device: 'str' = 'cpu')
```

Initialize the Box-wise Recall Loss. 

Parameters 
---------- 
- union_of_boxes (bool): Whether to use the union of boxes. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L629"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PixelWiseRecallLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L630"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: 'bool' = True, device: 'str' = 'cpu')
```

Initialize the Pixel-wise Recall Loss. 

Parameters 
---------- 
- union_of_boxes (bool): Whether to use the union of boxes. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L685"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxWisePrecisionLoss`
Box-wise PRECISION loss: 1 - mean(areas of the union of the boxes), 

This loss function calculates the recall loss based on the areas of the union of the predicted and true bounding boxes. The recall loss is defined as 1 minus the mean of the areas of the union of the boxes. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L692"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: 'bool' = True, device: 'str' = 'cpu')
```

Initialize the Box-wise Recall Loss. 

Parameters 
---------- 
- union_of_boxes (bool): Whether to use the union of boxes. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L756"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxWiseIoULoss`
Box-wise PRECISION loss: 1 - mean(areas of the union of the boxes), 

This loss function calculates the recall loss based on the areas of the union of the predicted and true bounding boxes. The recall loss is defined as 1 minus the mean of the areas of the union of the boxes. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L763"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: 'bool' = True, device: 'str' = 'cpu')
```

Initialize the Box-wise Recall Loss. 

Parameters 
---------- 
- union_of_boxes (bool): Whether to use the union of boxes. 

Returns 
------- 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L823"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NumberPredictionsGapLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L824"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(device: 'str' = 'cpu')
```

Initialize the Number Predictions Gap Loss. 

Returns 
------- 
- None 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
