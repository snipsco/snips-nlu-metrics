from __future__ import unicode_literals

import io
import json
import os
import zipfile
from abc import ABCMeta, abstractmethod
from builtins import object
from builtins import str
from copy import deepcopy

from future.utils import with_metaclass

from snips_nlu_metrics.utils.temp_utils import tempdir_ctx

TRAINED_ENGINE_FILENAME = "trained_assistant.json"


class Engine(with_metaclass(ABCMeta, object)):
    """
    Abstract class which represents an engine that can be used in the metrics
    API. All engine classes must inherit from `Engine`.
    """

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def parse(self, text):
        pass


def build_nlu_engine_class(training_class, inference_class,
                           training_config=None):
    _training_config = deepcopy(training_config)

    class NLUEngine(Engine):
        def __init__(self):
            self.inference_engine = None
            self.training_config = _training_config

        def fit(self, dataset):
            if self.training_config is not None:
                training_engine = training_class(config=self.training_config)
            else:
                training_engine = training_class()
            training_engine.fit(dataset)
            trained_engine_dict = training_engine.to_dict()
            self.inference_engine = get_inference_nlu_engine(
                trained_engine_dict, inference_class)

        def parse(self, text):
            return self.inference_engine.parse(text)

    return NLUEngine


def get_trained_nlu_engine(train_dataset, training_engine_class):
    language = train_dataset["language"]
    engine = training_engine_class(language)
    engine.fit(train_dataset)
    return engine


def get_inference_nlu_engine(trained_engine_dict, inference_engine_class):
    with tempdir_ctx() as engine_dir:
        trained_engine_path = os.path.join(engine_dir, TRAINED_ENGINE_FILENAME)
        archive_path = os.path.join(engine_dir, 'assistant.zip')

        with io.open(trained_engine_path, mode='w', encoding='utf8') as f:
            f.write(str(json.dumps(trained_engine_dict)))
        with zipfile.ZipFile(archive_path, 'w') as zf:
            zf.write(trained_engine_path, arcname=TRAINED_ENGINE_FILENAME)
        with io.open(archive_path, mode='rb') as f:
            data_zip = bytearray(f.read())

    return inference_engine_class(data_zip=data_zip)
