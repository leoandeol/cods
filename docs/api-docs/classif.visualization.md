<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/visualization.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `classif.visualization`
Visualization utilities for conformal classification predictions. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/classif/visualization.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_predictions`

```python
plot_predictions(
    idxs: list,
    preds: ClassificationPredictions,
    conf_cls: list = None
)
```

Plot classification predictions and optionally conformalized sets for given indices. 



**Args:**
 
---- 
 - <b>`idxs`</b> (list or int):  Indices of samples to plot. 
 - <b>`preds`</b> (ClassificationPredictions):  Predictions object containing image paths and labels. 
 - <b>`conf_cls`</b> (list, optional):  List of conformalized prediction sets. Defaults to None. 



**Raises:**
 
------ 
 - <b>`ValueError`</b>:  If lengths of predictions and conformalized sets do not match, or if class mapping is not set. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
