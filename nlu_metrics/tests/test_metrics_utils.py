from __future__ import unicode_literals

import unittest

from snips_nlu.constants import DATA, TEXT

from nlu_metrics.utils.metrics_utils import aggregate_metrics, \
    compute_utterance_metrics


class TestMetricsUtils(unittest.TestCase):
    def test_should_compute_utterance_metrics_when_wrong_intent(self):
        # Given
        text = "utterance of intent1"
        parsing = {
            "text": text,
            "intent": {
                "intentName": "intent2",
                "probability": 0.32
            },
            "slots": [
                {
                    "rawValue": "utterance",
                    "value": {
                        "kind": "Custom",
                        "value": "utterance"
                    },
                    "range": {
                        "start": 0,
                        "end": 9
                    },
                    "entity": "erroneous_entity",
                    "slotName": "erroneous_slot"
                }
            ]
        }
        utterance = {DATA: [{TEXT: text}]}
        intent_name = "intent1"
        # When
        metrics = compute_utterance_metrics(parsing, utterance, intent_name)
        # Then
        expected_metrics = {
            'intents': {
                'intent1': {
                    'false_negative': 1,
                    'false_positive': 0,
                    'true_positive': 0
                },
                'intent2': {
                    'false_negative': 0,
                    'false_positive': 1,
                    'true_positive': 0
                }
            },
            'slots': {
                'erroneous_slot': {
                    'false_negative': 0,
                    'false_positive': 0,
                    'true_positive': 0
                }
            }
        }
        self.assertDictEqual(expected_metrics, metrics)

    def test_aggregate_utils_should_work(self):
        # Given
        lhs_metrics = {
            "intents": {
                "intent1": {
                    "false_positive": 4,
                    "true_positive": 6,
                    "false_negative": 9
                },
                "intent2": {
                    "false_positive": 3,
                    "true_positive": 2,
                    "false_negative": 5
                },
            },
            "slots": {
                "slot1": {
                    "false_positive": 1,
                    "true_positive": 2,
                    "false_negative": 3
                },
                "slot2": {
                    "false_positive": 4,
                    "true_positive": 2,
                    "false_negative": 2
                },
            }
        }

        rhs_metrics = {
            "intents": {
                "intent1": {
                    "false_positive": 3,
                    "true_positive": 3,
                    "false_negative": 3
                },
                "intent2": {
                    "false_positive": 4,
                    "true_positive": 5,
                    "false_negative": 6
                },
                "intent3": {
                    "false_positive": 1,
                    "true_positive": 7,
                    "false_negative": 2
                },
            },
            "slots": {
                "slot1": {
                    "false_positive": 2,
                    "true_positive": 3,
                    "false_negative": 1
                },
            }
        }

        # When
        aggregated_metrics = aggregate_metrics(lhs_metrics, rhs_metrics)

        # Then
        expected_metrics = {
            "intents": {
                "intent1": {
                    "false_positive": 7,
                    "true_positive": 9,
                    "false_negative": 12
                },
                "intent2": {
                    "false_positive": 7,
                    "true_positive": 7,
                    "false_negative": 11
                },
                "intent3": {
                    "false_positive": 1,
                    "true_positive": 7,
                    "false_negative": 2
                },
            },
            "slots": {
                "slot1": {
                    "false_positive": 3,
                    "true_positive": 5,
                    "false_negative": 4
                },
                "slot2": {
                    "false_positive": 4,
                    "true_positive": 2,
                    "false_negative": 2
                },
            }
        }

        self.assertDictEqual(expected_metrics, aggregated_metrics)
