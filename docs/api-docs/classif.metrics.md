<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/metrics.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `classif.metrics`
Metrics for evaluating conformal classification predictions. 

**Global Variables**
---------------
- **CLASSIFICATION_LOSSES**

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/metrics.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_coverage`

```python
get_coverage(
    preds: ClassificationPredictions,
    conf_cls: Tensor,
    verbose: bool = True
)
```

Compute the coverage of the conformal prediction set. 



**Args:**
 
---- 
 - <b>`preds`</b> (ClassificationPredictions):  Predictions and ground truth of the classifier. 
 - <b>`conf_cls`</b> (torch.Tensor):  Conformalized predictions of the classifier. 
 - <b>`verbose`</b> (bool, optional):  Whether to print the coverage. Defaults to True. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Coverage indicator for each sample. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/metrics.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_empirical_risk`

```python
get_empirical_risk(
    preds: ClassificationPredictions,
    conf_cls: Tensor,
    loss: ClassificationLoss,
    verbose: bool = True
)
```

Compute the empirical risk of the conformal prediction set. 



**Args:**
 
---- 
 - <b>`preds`</b> (ClassificationPredictions):  Predictions and ground truth of the classifier. 
 - <b>`conf_cls`</b> (torch.Tensor):  Conformalized predictions of the classifier. 
 - <b>`loss`</b> (ClassificationLoss):  Loss function to use. 
 - <b>`verbose`</b> (bool, optional):  Whether to print the empirical risk. Defaults to True. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Empirical risk for each sample. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/metrics.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_empirical_safety`

```python
get_empirical_safety(
    preds: ClassificationPredictions,
    conf_cls: Tensor,
    loss: ClassificationLoss,
    verbose: bool = True
)
```

Compute the empirical safety of the conformal prediction set. 



**Args:**
 
---- 
 - <b>`preds`</b> (ClassificationPredictions):  Predictions and ground truth of the classifier. 
 - <b>`conf_cls`</b> (torch.Tensor):  Conformalized predictions of the classifier. 
 - <b>`loss`</b> (ClassificationLoss):  Loss function to use. 
 - <b>`verbose`</b> (bool, optional):  Whether to print the empirical safety. Defaults to True. 



**Returns:**
 
------- 
 - <b>`torch.Tensor`</b>:  Empirical safety for each sample. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
