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


def prefix_match(lhs_slot, rhs_slot):
    return lhs_slot.startswith(rhs_slot) or rhs_slot.startswith(lhs_slot)

metrics = compute_train_test_metrics(train_dataset="path/to/train_dataset.json", 
                                     test_dataset="path/to/test_dataset.json",
                                     training_engine_class=DumbTrainingEngine,
                                     inference_engine_class=DumbInferenceEngine,
                                     use_asr_output=True,
                                     slot_matching_lambda=prefix_match,
                                     verbose=True)
```

- `train_dataset`: dataset to use for training
- `test_dataset`: dataset to use for testing
- `training_engine_class`: NLU engine class to use for training
- `inference_engine_class`: NLU engine class to use for inference
- `use_asr_output`: bool (optional), whether the asr output should be
        used instead of utterance text
- `slot_matching_lambda`: optional function that specify how to match two slots. By default an exact match is used.
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
                                    train_size_ratio=0.5,
                                    use_asr_output=False,
                                    slot_matching_lambda=None)
```

- `dataset`: dataset to use during cross validation
- `training_engine_class`: NLU engine class to use for training
- `inference_engine_class`: NLU engine class to use for inference
- `nb_folds` (optional): number of folds to use, default to 5
- `train_size_ratio` (optional): proportion of utterances to use per intent for training, default to 1.0
- `use_asr_output`: bool (optional), whether the asr output should be
        used instead of utterance text
- `slot_matching_lambda`: optional function that specify how to match two slots. By default an exact match is used.
