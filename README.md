# NLU Metrics

[![Build Status](https://jenkins2.snips.ai/buildStatus/icon?job=SDK/asr-lm-adaptation/develop)](https://jenkins2.snips.ai/job/SDK/job/asr-lm-adaptation/view/Branches/job/develop)

Python package to compute NLU metrics

## Install
Requirements: Python2.7, [pip](https://pip.pypa.io/en/stable/installing/)

Create a `pip.conf` file with the following content (get user/password from @franblas): 
    
```config
[global]
index = https://<user>:<password>@nexus-repository.snips.ai/repository/pypi-internal/pypi
index-url = https://pypi.python.org/simple/
extra-index-url = https://<user>:<password>@nexus-repository.snips.ai/repository/pypi-internal/simple
```

Then:

```bash
virtualenv venv
# copy pip.conf to venv/
. venv/bin/activate
pip install nlu_metrics
```

## Metrics API

### Train/Test metrics

This API lets you train the model on a specific dataset and compute metrics another dataset:

```python
from nlu_metrics import compute_train_test_metrics

metrics = compute_train_test_metrics("path/to/train_dataset.json", 
                                     "path/to/test_dataset.json", 
                                     verbose=True)
```
`verbose=True` will output some logs about the model errors.
Optionally, you can specify the version of the training package (`snips_nlu_version`) and inference package (`snips_nlu_rust_version`). The `training_engine_class` parameter lets you use a specific NLU engine for training.

### Cross validation metrics

This API lets you compute metrics on a dataset using cross validation:

```python
from nlu_metrics import compute_cross_val_metrics

metrics = compute_cross_val_metrics("path/to/dataset.json")
```

Optionally, you can specify the version of the training package (`snips_nlu_version`) and inference package (`snips_nlu_rust_version`), as well as the max size of the training set (`max_utterances`) and the number of folds to use (`k_folds`).
The `training_engine_class` parameter lets you use a specific NLU engine for training.