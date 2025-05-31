<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.score`






---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(**kwargs)
```









---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(**kwargs)
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_margins`

```python
apply_margins(pred_boxes: Tensor) → Tensor
```

Applies margins to the predicted boxes. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The predicted boxes with applied margins. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_boxes: Tensor, quantile: float) → Tensor
```

Returns the set of boxes based on predicted boxes and quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The set of boxes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(image_shape: Tensor)
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L186"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_margins`

```python
apply_margins(pred_boxes: Tensor, quantile: float) → Tensor
```

Applies margins to the predicted boxes based on quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The predicted boxes with applied margins. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_boxes: Tensor, quantile: float) → Tensor
```

Returns the set of boxes based on predicted boxes and quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The set of boxes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UnionAdditiveSignedAssymetricHausdorffNCScore`
UnionAdditiveSignedAssymetricHausdorffNCScore is a class that calculates the score using the union additive signed asymmetric Hausdorff distance. 



**Args:**
 
----  None 



**Attributes:**
 
----------  None 

Methods: 
------- 
 - <b>`apply_margins`</b> (self, pred_boxes, quantile):  Applies margins to the predicted boxes based on quantile. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L250"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_margins`

```python
apply_margins(pred_boxes: Tensor, quantile: float) → Tensor
```

Applies margins to the predicted boxes based on quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The predicted boxes with applied margins. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_boxes: Tensor, quantile: float) → Tensor
```

Returns the set of boxes based on predicted boxes and quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The set of boxes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L279"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L297"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_margins`

```python
apply_margins(pred_boxes: Tensor, quantile: float) → Tensor
```

Applies margins to the predicted boxes based on quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The predicted boxes with applied margins. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_boxes: Tensor, quantile: float) → Tensor
```

Returns the set of boxes based on predicted boxes and quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The set of boxes. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L366"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UnionMultiplicativeSignedAssymetricHausdorffNCScore`
UnionMultiplicativeSignedAssymetricHausdorffNCScore is a class that calculates the score using the union multiplicative signed asymmetric Hausdorff distance. 



**Args:**
 
----  None 



**Attributes:**
 
----------  None 

Methods: 
------- 
 - <b>`apply_margins`</b> (self, pred_boxes, quantile):  Applies margins to the predicted boxes based on quantile. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L383"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L401"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_margins`

```python
apply_margins(pred_boxes: Tensor, quantile: float) → Tensor
```

Applies margins to the predicted boxes based on quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The predicted boxes with applied margins. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/score.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_set`

```python
get_set(pred_boxes: Tensor, quantile: float) → Tensor
```

Returns the set of boxes based on predicted boxes and quantile. 



**Args:**
 
---- 
 - <b>`pred_boxes`</b> (torch.Tensor):  Predicted boxes. 
 - <b>`quantile`</b> (float):  Quantile value. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  The set of boxes. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
