import io
import json
import os
import unittest

from mock import patch
from snips_nlu import SnipsNLUEngine
from snips_nlu_rust import NLUEngine as RustNLUEngine

from nlu_metrics import (compute_cross_val_metrics, compute_train_test_metrics)
from nlu_metrics.engine import build_nlu_engine_class
from nlu_metrics.metrics import compute_cross_val_nlu_metrics, \
    compute_train_test_nlu_metrics
from nlu_metrics.tests.engine_config import NLU_CONFIG
from nlu_metrics.utils.constants import METRICS, PARSING_ERRORS


class TestMetricsUtils(unittest.TestCase):
    @patch('nlu_metrics.metrics.compute_cross_val_metrics')
    def test_cross_val_nlu_metrics(self, mocked_cross_val_metrics):
        # Given
        mocked_metrics_result = {"metrics": "ok"}
        mocked_cross_val_metrics.return_value = mocked_metrics_result
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")
        # When/Then
        try:
            res = compute_cross_val_nlu_metrics(
                dataset=dataset_path, training_engine_class=SnipsNLUEngine,
                inference_engine_class=RustNLUEngine, nb_folds=5)
        except Exception as e:
            self.fail(e.message)

        self.assertDictEqual(mocked_metrics_result, res)

    def test_cross_val_metrics_should_skip_when_not_enough_data(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")

        # When
        result = compute_cross_val_nlu_metrics(
            dataset=dataset_path, training_engine_class=SnipsNLUEngine,
            inference_engine_class=RustNLUEngine, nb_folds=11)

        # Then
        expected_result = {
            METRICS: None,
            PARSING_ERRORS: []
        }
        self.assertDictEqual(expected_result, result)

    def test_end_to_end_cross_val_metrics(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")
        with io.open(dataset_path, encoding="utf8") as f:
            dataset = json.load(f)

        # When/Then
        try:
            engine_class = build_nlu_engine_class(SnipsNLUEngine,
                                                  RustNLUEngine)
            compute_cross_val_metrics(dataset=dataset,
                                      engine_class=engine_class, nb_folds=5)
        except Exception as e:
            self.fail(e.message)

    @patch('nlu_metrics.metrics.compute_train_test_metrics')
    def test_train_test_nlu_metrics(self, mocked_train_test_metrics):
        # Given
        mocked_metrics_result = {"metrics": "ok"}
        mocked_train_test_metrics.return_value = mocked_metrics_result
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")
        with io.open(dataset_path, encoding="utf8") as f:
            dataset = json.load(f)

        # When/Then
        try:
            res = compute_train_test_nlu_metrics(
                train_dataset=dataset, test_dataset=dataset,
                training_engine_class=SnipsNLUEngine,
                inference_engine_class=RustNLUEngine)
        except Exception as e:
            self.fail(e.message)

        self.assertDictEqual(mocked_metrics_result, res)

    def test_end_to_end_train_test_metrics(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")
        with io.open(dataset_path, encoding="utf8") as f:
            dataset = json.load(f)

        # When/Then
        try:
            engine_class = build_nlu_engine_class(SnipsNLUEngine,
                                                  RustNLUEngine)
            compute_train_test_metrics(
                train_dataset=dataset, test_dataset=dataset,
                engine_class=engine_class)
        except Exception as e:
            self.fail(e.message)

    def test_end_to_end_train_test_metrics_with_training_config(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")
        with io.open(dataset_path, encoding="utf8") as f:
            dataset = json.load(f)

        # When/Then
        try:
            engine_class = build_nlu_engine_class(SnipsNLUEngine,
                                                  RustNLUEngine,
                                                  training_config=NLU_CONFIG)
            compute_train_test_metrics(
                train_dataset=dataset, test_dataset=dataset,
                engine_class=engine_class)
        except Exception as e:
            self.fail(e.message)
