# NLU Metrics

[![Build Status](https://travis-ci.org/snipsco/snips-nlu-metrics.svg?branch=develop)](https://travis-ci.org/snipsco/snips-nlu-metrics)

Python package to compute metrics on a NLU/ASR parsing pipeline

## Install
 
```bash
pip install snips_nlu_metrics
```

## Pure NLU Metrics API

```python
from snips_nlu_metrics import (
    compute_train_test_nlu_metrics, compute_cross_val_nlu_metrics)
from snips_nlu import SnipsNLUEngine as NLUTrainingEngine
from snips_nlu_rust import NLUEngine as NLUInferenceEngine


tt_metrics = compute_train_test_nlu_metrics(train_dataset="path/to/train_dataset.json", 
                                            test_dataset="path/to/test_dataset.json",
                                            training_engine_class=NLUTrainingEngine,
                                            inference_engine_class=NLUInferenceEngine)

cv_metrics = compute_cross_val_nlu_metrics(dataset="path/to/dataset.json", 
                                           training_engine_class=NLUTrainingEngine,
                                           inference_engine_class=NLUInferenceEngine, 
                                           nb_folds=5)
```

## End-to-End Metrics API

The metrics API lets you compute metrics on a full end-to-end ASR + NLU pipeline.
To do that, you will need to implement an engine class that inherits or satisfy 
the API of the following `Engine` abstract class:

```python
from abc import ABCMeta, abstractmethod

class Engine(object):
    """
    Abstract class which represents an engine that can be used in the metrics
    API. All engine classes must inherit from `Engine` or satisfy its API.
    """
    __metaclass__ = ABCMeta

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
                                        engine_class=EndToEndEngine)

cv_metrics = compute_cross_val_metrics(dataset="path/to/dataset.json", 
                                       engine_class=EndToEndEngine, 
                                       nb_folds=5)
```