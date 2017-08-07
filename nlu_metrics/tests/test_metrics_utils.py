from __future__ import unicode_literals

import unittest

from snips_nlu.constants import DATA, TEXT, ENTITY, SLOT_NAME

from nlu_metrics.utils.metrics_utils import (aggregate_metrics,
                                             compute_utterance_metrics,
                                             compute_precision_recall)


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
            "intent1": {
                "intent": {
                    "false_negative": 1,
                    "false_positive": 0,
                    "true_positive": 0
                },
                "slots": {}
            },
            "intent2": {
                "intent": {
                    "false_negative": 0,
                    "false_positive": 1,
                    "true_positive": 0
                },
                "slots": {
                    "erroneous_slot": {
                        "false_negative": 0,
                        "false_positive": 0,
                        "true_positive": 0
                    }
                }
            }
        }

        self.assertDictEqual(expected_metrics, metrics)

    def test_should_compute_utterance_metrics_when_correct_intent(self):
        # Given
        text = "this is intent1 with slot1_value and slot2_value"
        intent_name = "intent1"
        parsing = {
            "text": text,
            "intent": {
                "intentName": intent_name,
                "probability": 0.32
            },
            "slots": [
                {
                    "rawValue": "slot1_value",
                    "value": {
                        "kind": "Custom",
                        "value": "slot1_value"
                    },
                    "range": {
                        "start": 21,
                        "end": 32
                    },
                    "entity": "entity1",
                    "slotName": "slot1"
                }
            ]
        }
        utterance = {
            DATA: [
                {
                    TEXT: "this is intent1 with "
                },
                {
                    TEXT: "slot1_value",
                    ENTITY: "entity1",
                    SLOT_NAME: "slot1"
                },
                {
                    TEXT: " and "
                },
                {
                    TEXT: "slot2_value",
                    ENTITY: "entity2",
                    SLOT_NAME: "slot2"
                }
            ]
        }
        # When
        metrics = compute_utterance_metrics(parsing, utterance, intent_name)
        # Then
        expected_metrics = {
            "intent1": {
                "intent": {
                    "false_negative": 0,
                    "false_positive": 0,
                    "true_positive": 1
                },
                "slots": {
                    "slot1": {
                        "false_negative": 0,
                        "false_positive": 0,
                        "true_positive": 1
                    },
                    "slot2": {
                        "false_negative": 1,
                        "false_positive": 0,
                        "true_positive": 0
                    }
                }
            }
        }

        self.assertDictEqual(expected_metrics, metrics)

    def test_aggregate_utils_should_work(self):
        # Given
        lhs_metrics = {
            "intent1": {
                "intent": {
                    "false_positive": 4,
                    "true_positive": 6,
                    "false_negative": 9
                },
                "slots": {
                    "slot1": {
                        "false_positive": 1,
                        "true_positive": 2,
                        "false_negative": 3
                    },
                },
            },
            "intent2": {
                "intent": {
                    "false_positive": 3,
                    "true_positive": 2,
                    "false_negative": 5
                },
                "slots": {
                    "slot2": {
                        "false_positive": 4,
                        "true_positive": 2,
                        "false_negative": 2
                    },
                }
            },
        }

        rhs_metrics = {
            "intent1": {
                "intent": {
                    "false_positive": 3,
                    "true_positive": 3,
                    "false_negative": 3
                },
                "slots": {
                    "slot1": {
                        "false_positive": 2,
                        "true_positive": 3,
                        "false_negative": 1
                    },
                }
            },
            "intent2": {
                "intent": {
                    "false_positive": 4,
                    "true_positive": 5,
                    "false_negative": 6
                },
                "slots": {}
            },
            "intent3": {
                "intent": {
                    "false_positive": 1,
                    "true_positive": 7,
                    "false_negative": 2
                },
                "slots": {}
            },
        }

        # When
        aggregated_metrics = aggregate_metrics(lhs_metrics, rhs_metrics)

        # Then
        expected_metrics = {
            "intent1": {
                "intent": {
                    "false_positive": 7,
                    "true_positive": 9,
                    "false_negative": 12,
                },
                "slots":
                    {
                        "slot1": {
                            "false_positive": 3,
                            "true_positive": 5,
                            "false_negative": 4
                        },
                    }
            },
            "intent2": {
                "intent": {
                    "false_positive": 7,
                    "true_positive": 7,
                    "false_negative": 11,
                },
                "slots": {
                    "slot2": {
                        "false_positive": 4,
                        "true_positive": 2,
                        "false_negative": 2
                    },
                }
            },
            "intent3": {
                "intent": {
                    "false_positive": 1,
                    "true_positive": 7,
                    "false_negative": 2
                },
                "slots": {}
            },
        }

        self.assertDictEqual(expected_metrics, aggregated_metrics)

    def test_should_compute_precision_and_recall(self):
        # Given
        metrics = {
            "intent1": {
                "intent": {
                    "false_positive": 7,
                    "true_positive": 9,
                    "false_negative": 12,
                },
                "slots":
                    {
                        "slot1": {
                            "false_positive": 3,
                            "true_positive": 5,
                            "false_negative": 4
                        },
                    }
            },
            "intent2": {
                "intent": {
                    "false_positive": 7,
                    "true_positive": 7,
                    "false_negative": 11,
                },
                "slots": {
                    "slot2": {
                        "false_positive": 4,
                        "true_positive": 2,
                        "false_negative": 2
                    },
                }
            },
        }

        # When
        augmented_metrics = compute_precision_recall(metrics)

        # Then
        expected_metrics = {
            "intent1": {
                "intent": {
                    "false_positive": 7,
                    "true_positive": 9,
                    "false_negative": 12,
                    "precision": 9. / (7. + 9.),
                    "recall": 9. / (12. + 9.),

                },
                "slots":
                    {
                        "slot1": {
                            "false_positive": 3,
                            "true_positive": 5,
                            "false_negative": 4,
                            "precision": 5. / (5. + 3.),
                            "recall": 5. / (5. + 4.),
                        },
                    }
            },
            "intent2": {
                "intent": {
                    "false_positive": 7,
                    "true_positive": 7,
                    "false_negative": 11,
                    "precision": 7. / (7. + 7.),
                    "recall": 7. / (7. + 11.),
                },
                "slots": {
                    "slot2": {
                        "false_positive": 4,
                        "true_positive": 2,
                        "false_negative": 2,
                        "precision": 2. / (2. + 4.),
                        "recall": 2. / (2. + 2.),
                    },
                }
            },
        }
        self.assertDictEqual(expected_metrics, augmented_metrics)
