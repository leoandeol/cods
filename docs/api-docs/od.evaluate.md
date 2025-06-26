<!-- markdownlint-disable -->

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/evaluate.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `od.evaluate`
Evaluation functionality for object detection models with conformal prediction. 

This module provides comprehensive evaluation capabilities for object detection models using conformal prediction, including hyperparameter tuning, model comparison, and performance metric computation across different evaluation modes. 

**Global Variables**
---------------
- **MODES**

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/evaluate.py#L338"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parse_args`

```python
parse_args()
```

Parse command line arguments for the benchmark. 



**Returns:**
 
 - <b>`argparse.Namespace`</b>:  Parsed command line arguments. 


---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/evaluate.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Benchmark`
Benchmark class for evaluating object detection models with conformal prediction. 

Provides comprehensive benchmarking capabilities including hyperparameter sweeps, model comparison, and performance evaluation across different conformal prediction configurations. 

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/evaluate.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(config, device)
```

Initialize the benchmark with configuration and device. 



**Args:**
 
 - <b>`config`</b> (dict):  Configuration dictionary containing benchmark parameters. 
 - <b>`device`</b> (str):  Device to use for computation ('cpu' or 'cuda'). 




---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/evaluate.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(threads=1)
```

Run the benchmark experiments. 



**Args:**
 
 - <b>`threads`</b> (int, optional):  Number of threads to use. Defaults to 1.  Multithreading is not yet implemented. 



**Raises:**
 
 - <b>`NotImplementedError`</b>:  If threads > 1 as multithreading is not implemented. 

---

<a href="https://github.com/leoandeol/cods/blob/main/cods/od/evaluate.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run_experiment`

```python
run_experiment(experiment, verbose=False)
```

Run a single experiment with the given configuration. 



**Args:**
 
 - <b>`experiment`</b> (dict):  Experiment configuration containing all parameters. 
 - <b>`verbose`</b> (bool, optional):  Whether to log detailed information. Defaults to False. 



**Returns:**
 
 - <b>`dict`</b>:  Experiment results including metrics and configuration. 



**Raises:**
 
 - <b>`NotImplementedError`</b>:  If dataset or model is not implemented. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
