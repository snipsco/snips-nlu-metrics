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

In order to compute metrics, you will need to implement an engine class that inherits from the following `Engine` abstract class:

```python
from abc import ABCMeta, abstractmethod

class Engine(object):
    """
    Abstract class which represents an engine that can be used in the metrics
    API. All engine classes must inherit from `Engine`.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, language):
        pass

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def parse(self, text):
        pass
``` 

If you intend to compute pure NLU metrics, you can use the following helper to build this engine class (provided you have install `snips_nlu` and `snips_nlu_rust` packages):

```python
from nlu_metrics import build_nlu_engine_class
from snips_nlu import SnipsNLUEngine as NLUTrainingEngine
from snips_nlu_rust import NLUEngine as NLUInferenceEngine

engine_class = build_nlu_engine_class(NLUTrainingEngine, NLUInferenceEngine)
```

For more sophisticated use cases, you will have to create your own custom engine class.

### Train/Test metrics

This API lets you train the model on a specific dataset and compute metrics another dataset:

```python
from nlu_metrics import compute_train_test_metrics, build_nlu_engine_class
from snips_nlu import SnipsNLUEngine as NLUTrainingEngine
from snips_nlu_rust import NLUEngine as NLUInferenceEngine

engine_class = build_nlu_engine_class(NLUTrainingEngine, NLUInferenceEngine)

def prefix_match(lhs_slot, rhs_slot):
    """Example of a custom slot matching function based on prefix"""
    return lhs_slot.startswith(rhs_slot) or rhs_slot.startswith(lhs_slot)

metrics = compute_train_test_metrics(train_dataset="path/to/train_dataset.json", 
                                     test_dataset="path/to/test_dataset.json",
                                     engine_class=engine_class,
                                     slot_matching_lambda=prefix_match)
```

- `train_dataset`: dataset to use for training
- `test_dataset`: dataset to use for testing
- `engine_class`: engine class to use for training and inference, must inherit from `Engine`
- `slot_matching_lambda`: optional function that specifies how to match two slots. By default, exact match is used.

### Cross validation metrics

This API lets you compute metrics on a dataset using cross validation, here is how you can use (provided you have installed `snips_nlu` and `snips_nlu_rust`):

```python
from nlu_metrics import compute_cross_val_metrics, build_nlu_engine_class
from snips_nlu import SnipsNLUEngine as NLUTrainingEngine
from snips_nlu_rust import NLUEngine as NLUInferenceEngine

engine_class = build_nlu_engine_class(NLUTrainingEngine, NLUInferenceEngine)

metrics = compute_cross_val_metrics(dataset="path/to/dataset.json",
                                    engine_class=engine_class,
                                    nb_folds=5,
                                    train_size_ratio=0.5,
                                    slot_matching_lambda=None)
```

- `dataset`: dataset to use during cross validation
- `engine_class`: engine class to use for training and inference, must inherit from `Engine`
- `nb_folds` (optional): number of folds to use, default to 5
- `train_size_ratio` (optional): proportion of utterances to use per intent for training, default to 1.0
- `slot_matching_lambda`: optional function that specifies how to match two slots. By default, exact match is used.
