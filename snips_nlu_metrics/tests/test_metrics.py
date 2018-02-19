import io
import json
import os
import unittest

from mock import patch

from snips_nlu_metrics.engine import build_nlu_engine_class
from snips_nlu_metrics.metrics import (compute_cross_val_metrics,
                                       compute_train_test_metrics,
                                       compute_cross_val_nlu_metrics,
                                       compute_train_test_nlu_metrics)
from snips_nlu_metrics.tests.engine_config import NLU_CONFIG
from snips_nlu_metrics.tests.mock_engine import (MockTrainingEngine,
                                                 MockInferenceEngine)
from snips_nlu_metrics.utils.constants import METRICS, PARSING_ERRORS


class TestMetrics(unittest.TestCase):
    def test_cross_val_nlu_metrics(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")
        # When
        try:
            res = compute_cross_val_nlu_metrics(
                dataset=dataset_path, training_engine_class=MockTrainingEngine,
                inference_engine_class=MockInferenceEngine, nb_folds=2)
        except Exception as e:
            self.fail(e.args[0])

        # Then
        expected_metrics = {
            'null': {
                'intent': {
                    'true_positive': 0,
                    'false_positive': 11,
                    'false_negative': 0,
                    'precision': 0.0,
                    'recall': 0.0
                },
                'slots': {},
                'intent_utterances': 0
            },
            'MakeCoffee': {
                'intent': {
                    'true_positive': 0,
                    'false_positive': 0,
                    'false_negative': 7,
                    'precision': 0.0,
                    'recall': 0.0
                },
                'slots': {
                    'number_of_cups': {
                        'true_positive': 0,
                        'false_positive': 0,
                        'false_negative': 0,
                        'precision': 0.0,
                        'recall': 0.0
                    }
                },
                'intent_utterances': 7
            },
            'MakeTea': {
                'intent': {
                    'true_positive': 0,
                    'false_positive': 0,
                    'false_negative': 4,
                    'precision': 0.0,
                    'recall': 0.0
                },
                'slots': {
                    'number_of_cups': {
                        'true_positive': 0,
                        'false_positive': 0,
                        'false_negative': 0,
                        'precision': 0.0,
                        'recall': 0.0
                    },
                    'beverage_temperature': {
                        'true_positive': 0,
                        'false_positive': 0,
                        'false_negative': 0,
                        'precision': 0.0,
                        'recall': 0.0
                    }
                },
                'intent_utterances': 4
            }
        }

        self.assertDictEqual(expected_metrics, res["metrics"])

    def test_cross_val_metrics_should_skip_when_not_enough_data(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")

        # When
        result = compute_cross_val_nlu_metrics(
            dataset=dataset_path, training_engine_class=MockTrainingEngine,
            inference_engine_class=MockInferenceEngine, nb_folds=11)

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
            engine_class = build_nlu_engine_class(MockTrainingEngine,
                                                  MockInferenceEngine)
            compute_cross_val_metrics(dataset=dataset,
                                      engine_class=engine_class, nb_folds=5)
        except Exception as e:
            self.fail(e.args[0])

    @patch('snips_nlu_metrics.metrics.compute_train_test_metrics')
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
                training_engine_class=MockTrainingEngine,
                inference_engine_class=MockInferenceEngine)
        except Exception as e:
            self.fail(e.args[0])

        self.assertDictEqual(mocked_metrics_result, res)

    def test_end_to_end_train_test_metrics(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")
        with io.open(dataset_path, encoding="utf8") as f:
            dataset = json.load(f)

        # When/Then
        try:
            engine_class = build_nlu_engine_class(MockTrainingEngine,
                                                  MockInferenceEngine)
            compute_train_test_metrics(
                train_dataset=dataset, test_dataset=dataset,
                engine_class=engine_class)
        except Exception as e:
            self.fail(e.args[0])

    def test_end_to_end_train_test_metrics_with_training_config(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")
        with io.open(dataset_path, encoding="utf8") as f:
            dataset = json.load(f)

        # When/Then
        try:
            engine_class = build_nlu_engine_class(MockTrainingEngine,
                                                  MockInferenceEngine,
                                                  training_config=NLU_CONFIG)
            compute_train_test_metrics(
                train_dataset=dataset, test_dataset=dataset,
                engine_class=engine_class)
        except Exception as e:
            self.fail(e.args[0])
