<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/visualization.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.visualization`





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/visualization.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_preds`

```python
plot_preds(
    preds,
    idx,
    conf_boxes: list,
    conf_cls: list,
    confidence_threshold=None,
    save_as=None
)
```

Plot the predictions of an object detection model. 



**Args:**
 
 - <b>`preds`</b> (object):  Object containing the predictions. 
 - <b>`idx`</b> (int):  Index of the image to plot. 
 - <b>`conf_boxes`</b> (list):  List of confidence boxes. 
 - <b>`conf_cls`</b> (list):  List of confidence classes. 
 - <b>`confidence_threshold`</b> (float, optional):  Confidence threshold for filtering predictions. If not provided, the threshold from `preds` will be used. Defaults to None. 
 - <b>`save_as`</b> (str, optional):  File path to save the plot. Defaults to None. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
