import json
import unittest

import io

import os
from snips_nlu import SnipsNLUEngine
from snips_nlu_rust import NLUEngine as RustNLUEngine

from nlu_metrics import compute_cross_val_metrics, compute_train_test_metrics


class TestMetricsUtils(unittest.TestCase):
    def test_end_to_end_cross_val_metrics(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")
        with io.open(dataset_path, encoding="utf8") as f:
            dataset = json.load(f)

        # When/Then
        try:
            compute_cross_val_metrics(dataset=dataset,
                                      training_engine_class=SnipsNLUEngine,
                                      inference_engine_class=RustNLUEngine,
                                      nb_folds=5)
        except Exception as e:
            self.fail(e.message)

    def test_end_to_end_train_test_metrics(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")
        with io.open(dataset_path, encoding="utf8") as f:
            dataset = json.load(f)

        # When/Then
        try:
            compute_train_test_metrics(train_dataset=dataset,
                                       test_dataset=dataset,
                                       training_engine_class=SnipsNLUEngine,
                                       inference_engine_class=RustNLUEngine)
        except Exception as e:
            self.fail(e.message)
