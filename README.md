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

class DumbTrainingEngine(object):
    def __init__(self, language):
        self.language = language

    def fit(self, dataset):
        pass

    def to_dict(self):
        return dict()


class DumbInferenceEngine(object):
    def __init__(self, language, data_zip):
        pass

    def parse(self, text):
        return {"text": text, "intent": None, "slots": None}

metrics = compute_train_test_metrics(train_dataset="path/to/train_dataset.json", 
                                     test_dataset="path/to/test_dataset.json",
                                     training_engine_class=DumbTrainingEngine,
                                     inference_engine_class=DumbInferenceEngine,
                                     verbose=True)
```

- `snips_nlu_version`: version of the training package
- `snips_nlu_rust_version`: version of the inference package
- `training_engine_class` (optional): specific NLU engine class to use for training
- `verbose` (optional): if `True`, will output some logs about the model errors.

### Cross validation metrics

This API lets you compute metrics on a dataset using cross validation, here is how you can use (provided you have installed `snips_nlu` and `snips_nlu_rust`):

```python
from nlu_metrics import compute_cross_val_metrics
from snips_nlu import SnipsNLUEngine
from snips_nlu_rust import NLUEngine as RustNLUEngine

metrics = compute_cross_val_metrics(dataset="path/to/dataset.json",
                                    training_engine_class=SnipsNLUEngine,
                                    inference_engine_class=RustNLUEngine,
                                    nb_folds=5,
                                    training_utterances=50)
```
- `snips_nlu_version`: version of the training package
- `snips_nlu_rust_version`: version of the inference package
- `training_engine_class` (optional): specific NLU engine class to use for training
- `max_utterances` (optional): max size of the training set, default to the size of the dataset
- `nb_folds` (optional): number of folds to use, default to 5