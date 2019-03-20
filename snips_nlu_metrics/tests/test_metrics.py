import io
import json
import os
import unittest

from snips_nlu_metrics.metrics import (compute_cross_val_metrics,
                                       compute_train_test_metrics)
from snips_nlu_metrics.tests.mock_engine import MockEngine
from snips_nlu_metrics.utils.constants import METRICS, PARSINGS, \
    CONFUSION_MATRIX, AVERAGE_METRICS


class TestMetrics(unittest.TestCase):
    def test_compute_cross_val_metrics(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")
        with io.open(dataset_path, encoding="utf8") as f:
            dataset = json.load(f)

        # When/Then
        try:
            res = compute_cross_val_metrics(
                dataset=dataset, engine_class=MockEngine, nb_folds=2)
        except Exception as e:
            self.fail(e.args[0])

        expected_metrics = {
            "null": {
                "intent": {
                    "true_positive": 0,
                    "false_positive": 11,
                    "false_negative": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                },
                "exact_parsings": 0,
                "slots": {},
                "intent_utterances": 0
            },
            "MakeCoffee": {
                "intent": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 7,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                },
                "exact_parsings": 0,
                "slots": {
                    "number_of_cups": {
                        "true_positive": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0
                    }
                },
                "intent_utterances": 7
            },
            "MakeTea": {
                "intent": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 4,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                },
                "exact_parsings": 0,
                "slots": {
                    "number_of_cups": {
                        "true_positive": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0
                    },
                    "beverage_temperature": {
                        "true_positive": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0
                    }
                },
                "intent_utterances": 4
            }
        }

        self.assertDictEqual(expected_metrics, res["metrics"])

        # Testing parallel CV with various number of workers
        try:
            res = compute_cross_val_metrics(
                dataset=dataset, engine_class=MockEngine, nb_folds=2,
                num_workers=4)
        except Exception as e:
            self.fail(e.args[0])
        self.assertDictEqual(expected_metrics, res["metrics"])

        try:
            res = compute_cross_val_metrics(
                dataset=dataset, engine_class=MockEngine, nb_folds=2,
                num_workers=2)
        except Exception as e:
            self.fail(e.args[0])
        self.assertDictEqual(expected_metrics, res["metrics"])

    def test_compute_cross_val_metrics_without_slot_metrics(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")
        with io.open(dataset_path, encoding="utf8") as f:
            dataset = json.load(f)

        # When/Then
        try:
            res = compute_cross_val_metrics(
                dataset=dataset, engine_class=MockEngine, nb_folds=2,
                include_slot_metrics=False)
        except Exception as e:
            self.fail(e.args[0])

        expected_metrics = {
            "null": {
                "intent": {
                    "true_positive": 0,
                    "false_positive": 11,
                    "false_negative": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                },
                "intent_utterances": 0,
                "exact_parsings": 0
            },
            "MakeCoffee": {
                "intent": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 7,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                },
                "intent_utterances": 7,
                "exact_parsings": 0
            },
            "MakeTea": {
                "intent": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 4,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                },
                "intent_utterances": 4,
                "exact_parsings": 0
            }
        }

        self.assertDictEqual(expected_metrics, res["metrics"])

    def test_cross_val_metrics_should_skip_when_not_enough_data(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")

        # When
        result = compute_cross_val_metrics(
            dataset=dataset_path, engine_class=MockEngine, nb_folds=11)

        # Then
        expected_result = {
            AVERAGE_METRICS: None,
            CONFUSION_MATRIX: None,
            METRICS: None,
            PARSINGS: []
        }
        self.assertDictEqual(expected_result, result)

    def test_compute_train_test_metrics(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")
        with io.open(dataset_path, encoding="utf8") as f:
            dataset = json.load(f)

        # When/Then
        try:
            res = compute_train_test_metrics(
                train_dataset=dataset, test_dataset=dataset,
                engine_class=MockEngine)
        except Exception as e:
            self.fail(e.args[0])

        expected_metrics = {
            "MakeCoffee": {
                "intent": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 7,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                },
                "slots": {
                    "number_of_cups": {
                        "true_positive": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0
                    }
                },
                "intent_utterances": 7,
                "exact_parsings": 0,
            },
            "null": {
                "intent": {
                    "true_positive": 0,
                    "false_positive": 11,
                    "false_negative": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0},
                "slots": {},
                "intent_utterances": 0,
                "exact_parsings": 0,
            }, "MakeTea": {
                "intent": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 4,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                },
                "slots": {
                    "number_of_cups": {
                        "true_positive": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0
                    },
                    "beverage_temperature": {
                        "true_positive": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0
                    }
                },
                "intent_utterances": 4,
                "exact_parsings": 0,
            }
        }

        self.assertDictEqual(expected_metrics, res["metrics"])

    def test_compute_train_test_metrics_without_slots_metrics(self):
        # Given
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "resources", "beverage_dataset.json")
        with io.open(dataset_path, encoding="utf8") as f:
            dataset = json.load(f)

        # When/Then
        try:
            res = compute_train_test_metrics(
                train_dataset=dataset, test_dataset=dataset,
                engine_class=MockEngine, include_slot_metrics=False)
        except Exception as e:
            self.fail(e.args[0])

        expected_metrics = {
            "MakeCoffee": {
                "intent": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 7,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                },
                "intent_utterances": 7,
                "exact_parsings": 0,
            },
            "null": {
                "intent": {
                    "true_positive": 0,
                    "false_positive": 11,
                    "false_negative": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0},
                "intent_utterances": 0,
                "exact_parsings": 0,
            }, "MakeTea": {
                "intent": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 4,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                },
                "intent_utterances": 4,
                "exact_parsings": 0,
            }
        }

        self.assertDictEqual(expected_metrics, res["metrics"])
