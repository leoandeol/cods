<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.score`
Non-conformity scoring functions for object detection conformal prediction. 

This module provides non-conformity scoring classes for object detection tasks, including objectness scoring and other metrics used in the conformal prediction framework to quantify prediction uncertainty. 



---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ObjectnessNCScore`
ObjectnessNCScore is a class that calculates the score for objectness prediction. 



**Args:**
 
---- 
 - <b>`kwargs`</b>:  Additional keyword arguments. 



**Attributes:**
 
---------- None 

Methods: 
------- 
 - <b>`__call__`</b> (self, n_gt, confidence):  Calculates the score based on the number of ground truth objects and confidence values. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(**kwargs)
```

Initialize the NCScore. 



**Args:**
 
 - <b>`**kwargs`</b>:  Additional keyword arguments. 





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ODNCScore`
ODNCScore is an abstract class for calculating the score in object detection tasks. 



**Args:**
 
---- 
 - <b>`kwargs`</b>:  Additional keyword arguments. 



**Attributes:**
 
---------- None 

Methods: 
------- 
 - <b>`__call__`</b> (self, pred_boxes, true_box, **kwargs):  Calculates the score based on predicted boxes and true box. 
 - <b>`get_set`</b> (self, pred_boxes, quantile):  Returns the set of boxes based on predicted boxes and quantile. 
 - <b>`apply_margins`</b> (self, pred_boxes):  Applies margins to the predicted boxes. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(**kwargs)
```

Initialize the ODNCScore. 



**Args:**
 
 - <b>`**kwargs`</b>:  Additional keyword arguments. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_margins`

```python
apply_margins(pred_boxes: Tensor) → Tensor
```

Apply margins to the predicted boxes. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The predicted boxes with applied margins. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_boxes: Tensor, quantile: float) → Tensor
```

Return the set of boxes based on predicted boxes and quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The set of boxes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MinAdditiveSignedAssymetricHausdorffNCScore`
MinAdditiveSignedAssymetricHausdorffNCScore is a class that calculates the score using the minimum additive signed asymmetric Hausdorff distance. 



**Args:**
 
---- 
 - <b>`image_shape`</b> (torch.Tensor, optional):  The shape of the image. Defaults to None. 



**Attributes:**
 
---------- 
 - <b>`image_shape`</b> (torch.Tensor):  The shape of the image. 

Methods: 
------- 
 - <b>`__call__`</b> (self, pred_boxes, true_box):  Calculates the score based on predicted boxes and true box. 
 - <b>`apply_margins`</b> (self, pred_boxes, quantile):  Applies margins to the predicted boxes based on quantile. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(image_shape: Tensor)
```

Initialize the IoUNCScore with image shape. 



**Args:**
 
 - <b>`image_shape`</b> (torch.Tensor):  The shape of the image. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_margins`

```python
apply_margins(pred_boxes: Tensor, quantile: float) → Tensor
```

Apply margins to the predicted boxes based on quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The predicted boxes with applied margins. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_boxes: Tensor, quantile: float) → Tensor
```

Return the set of boxes based on predicted boxes and quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The set of boxes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L243"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UnionAdditiveSignedAssymetricHausdorffNCScore`
UnionAdditiveSignedAssymetricHausdorffNCScore is a class that calculates the score using the union additive signed asymmetric Hausdorff distance. 



**Args:**
 
----  None 



**Attributes:**
 
----------  None 

Methods: 
------- 
 - <b>`apply_margins`</b> (self, pred_boxes, quantile):  Applies margins to the predicted boxes based on quantile. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L260"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the GIoUNCScore. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L279"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_margins`

```python
apply_margins(pred_boxes: Tensor, quantile: float) → Tensor
```

Apply margins to the predicted boxes based on quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The predicted boxes with applied margins. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_boxes: Tensor, quantile: float) → Tensor
```

Return the set of boxes based on predicted boxes and quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The set of boxes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MinMultiplicativeSignedAssymetricHausdorffNCScore`
MinMultiplicativeSignedAssymetricHausdorffNCScore is a class that calculates the score using the minimum multiplicative signed asymmetric Hausdorff distance. 



**Args:**
 
----  None 



**Attributes:**
 
----------  None 

Methods: 
------- 
 - <b>`__call__`</b> (self, pred_boxes, true_box):  Calculates the score based on predicted boxes and true box. 
 - <b>`apply_margins`</b> (self, pred_boxes, quantile):  Applies margins to the predicted boxes based on quantile. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L327"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the GIoUNCScore. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L366"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_margins`

```python
apply_margins(pred_boxes: Tensor, quantile: float) → Tensor
```

Apply margins to the predicted boxes based on quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The predicted boxes with applied margins. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_boxes: Tensor, quantile: float) → Tensor
```

Return the set of boxes based on predicted boxes and quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The set of boxes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L398"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UnionMultiplicativeSignedAssymetricHausdorffNCScore`
UnionMultiplicativeSignedAssymetricHausdorffNCScore is a class that calculates the score using the union multiplicative signed asymmetric Hausdorff distance. 



**Args:**
 
----  None 



**Attributes:**
 
----------  None 

Methods: 
------- 
 - <b>`apply_margins`</b> (self, pred_boxes, quantile):  Applies margins to the predicted boxes based on quantile. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L415"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the GIoUNCScore. 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L434"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_margins`

```python
apply_margins(pred_boxes: Tensor, quantile: float) → Tensor
```

Apply margins to the predicted boxes based on quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The predicted boxes with applied margins. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_boxes: Tensor, quantile: float) → Tensor
```

Return the set of boxes based on predicted boxes and quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The set of boxes. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
