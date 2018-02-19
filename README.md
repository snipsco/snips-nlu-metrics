# NLU Metrics

Python package to compute metrics on a NLU/ASR parsing pipeline

## Install
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

## Pure NLU Metrics API

```python
from snips_nlu_metrics import (
    compute_train_test_nlu_metrics, compute_cross_val_nlu_metrics)
from snips_nlu import SnipsNLUEngine as NLUTrainingEngine
from snips_nlu_rust import NLUEngine as NLUInferenceEngine


def prefix_match(lhs_slot, rhs_slot):
    """Example of a custom slot matching function based on prefix"""
    expected_slot_text = lhs_slot["text"]
    parsed_slot_text = rhs_slot["rawValue"]
    return expected_slot_text.startswith(parsed_slot_text) or \
        parsed_slot_text.startswith(expected_slot_text)

tt_metrics = compute_train_test_nlu_metrics(train_dataset="path/to/train_dataset.json", 
                                            test_dataset="path/to/test_dataset.json",
                                            training_engine_class=NLUTrainingEngine,
                                            inference_engine_class=NLUInferenceEngine,
                                            slot_matching_lambda=prefix_match)

cv_metrics = compute_cross_val_nlu_metrics(dataset="path/to/dataset.json", 
                                           training_engine_class=NLUTrainingEngine,
                                           inference_engine_class=NLUInferenceEngine, 
                                           nb_folds=5, 
                                           train_size_ratio=0.5,
                                           drop_entities=False,
                                           slot_matching_lambda=None)
```

## End-to-End Metrics API

The metrics API lets you compute metrics on a full end-to-end ASR + NLU pipeline.
To do that, you will need to implement an engine class that inherits from the following `Engine` abstract class:

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

Here is how you can use the end-to-end metrics API, if you have a `EndToEndEngine` that inherits from `Engine`:

```python
from snips_nlu_metrics import compute_train_test_metrics, compute_cross_val_metrics


tt_metrics = compute_train_test_metrics(train_dataset="path/to/train_dataset.json", 
                                        test_dataset="path/to/test_dataset.json",
                                        engine_class=EndToEndEngine,
                                        slot_matching_lambda=None)

cv_metrics = compute_cross_val_metrics(dataset="path/to/dataset.json", 
                                       engine_class=EndToEndEngine, 
                                       nb_folds=5, 
                                       train_size_ratio=0.5,
                                       drop_entities=False,
                                       slot_matching_lambda=None)
```