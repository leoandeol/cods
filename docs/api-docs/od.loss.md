<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.loss`






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(upper_bound: int, **kwargs)
```

Initialize the Object Detection Loss. 



**Parameters:**
 
- upper_bound (int): The upper bound of the loss. 



**Returns:**
 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ObjectnessLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(upper_bound: int = 1, **kwargs)
```

Initialize the Objectness Loss. 



**Parameters:**
 
- upper_bound (int): The upper bound of the loss. 



**Returns:**
 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassificationLossWrapper`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(classification_loss, **kwargs)
```

Initialize the Classification Loss Wrapper. 



**Parameters:**
 
- classification_loss (Loss): The classification loss. 



**Returns:**
 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MaximumLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*losses)
```

Initialize the Maximum Loss. 



**Parameters:**
 
- losses (list): The list of losses. 



**Returns:**
 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HausdorffSignedDistanceLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(beta: float = 0.25)
```

Initialize the Hausdorff Signed Distance Loss. 



**Parameters:**
 
- beta (float): The beta value. 



**Returns:**
 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L177"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClassBoxWiseRecallLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: bool = True)
```

Initialize the Box-wise Recall Loss. 



**Parameters:**
 
- union_of_boxes (bool): Whether to use the union of boxes. 



**Returns:**
 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L231"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BoxWiseRecallLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: bool = True)
```

Initialize the Box-wise Recall Loss. 



**Parameters:**
 
- union_of_boxes (bool): Whether to use the union of boxes. 



**Returns:**
 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L278"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PixelWiseRecallLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L279"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(union_of_boxes: bool = True)
```

Initialize the Pixel-wise Recall Loss. 



**Parameters:**
 
- union_of_boxes (bool): Whether to use the union of boxes. 



**Returns:**
 
- None 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L320"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NumberPredictionsGapLoss`




<a href="https://github.com/leoandeol/cods/blob/main/cods/od/loss.py#L321"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the Number Predictions Gap Loss. 



**Returns:**
 
- None 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
