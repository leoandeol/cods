<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/visualization.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.visualization`





---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/visualization.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_preds`

```python
plot_preds(
    idx,
    predictions: ODPredictions,
    conformalized_predictions: ODConformalizedPredictions = None,
    confidence_threshold=None,
    idx_to_label: dict = None,
    save_as=None
)
```

Plot the predictions of an object detection model. 



**Args:**
 
---- 
 - <b>`preds`</b> (object):  Object containing the predictions. 
 - <b>`idx`</b> (int):  Index of the image to plot. 
 - <b>`conf_boxes`</b> (list):  List of confidence boxes. 
 - <b>`conf_cls`</b> (list):  List of confidence classes. 
 - <b>`confidence_threshold`</b> (float, optional):  Confidence threshold for filtering predictions. If not provided, the threshold from `preds` will be used. Defaults to None. 
 - <b>`save_as`</b> (str, optional):  File path to save the plot. Defaults to None. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/visualization.py#L177"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_pdf_with_plots`

```python
create_pdf_with_plots(
    predictions: ODPredictions,
    conformalized_predictions: ODConformalizedPredictions = None,
    confidence_threshold=None,
    idx_to_label: dict = None,
    output_pdf='output.pdf'
)
```

Create a PDF with plots for each image in the predictions. 



**Args:**
 
---- 
 - <b>`predictions`</b> (ODPredictions):  Object containing the predictions. 
 - <b>`conformalized_predictions`</b> (ODConformalizedPredictions, optional):  Object containing conformalized predictions. Defaults to None. 
 - <b>`confidence_threshold`</b> (float, optional):  Confidence threshold for filtering predictions. Defaults to None. 
 - <b>`idx_to_label`</b> (dict, optional):  Mapping from class indices to labels. Defaults to None. 
 - <b>`output_pdf`</b> (str, optional):  Path to save the output PDF. Defaults to "output.pdf". 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/visualization.py#L340"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_histograms_predictions`

```python
plot_histograms_predictions(predictions: ODPredictions)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
