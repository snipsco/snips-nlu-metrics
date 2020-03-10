import pytest

from snips_nlu_metrics.metrics import (
    compute_cross_val_metrics,
    compute_train_test_metrics,
)
from snips_nlu_metrics.tests.mock_engine import (
    MockEngine,
    MockEngineSegfault,
    KeyWordMatchingEngine,
)
from snips_nlu_metrics.utils.constants import (
    METRICS,
    PARSING_ERRORS,
    CONFUSION_MATRIX,
    AVERAGE_METRICS,
)


def test_compute_cross_val_metrics(logger, beverage_dataset):
    # When/Then
    try:
        res = compute_cross_val_metrics(
            dataset=beverage_dataset, engine_class=MockEngine, nb_folds=2
        )
    except Exception as e:
        raise AssertionError(e.args[0])

    expected_metrics = {
        "null": {
            "intent": {
                "true_positive": 0,
                "false_positive": 11,
                "false_negative": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "exact_parsings": 0,
            "slots": {},
            "intent_utterances": 0,
        },
        "MakeCoffee": {
            "intent": {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 7,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "exact_parsings": 0,
            "slots": {
                "number_of_cups": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
            },
            "intent_utterances": 7,
        },
        "MakeTea": {
            "intent": {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 4,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "exact_parsings": 0,
            "slots": {
                "number_of_cups": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                },
                "beverage_temperature": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                },
            },
            "intent_utterances": 4,
        },
    }

    assert expected_metrics, res["metrics"]


def test_compute_cross_val_metrics_with_intents_filter(
    logger, keyword_matching_dataset
):
    # When/Then
    res = compute_cross_val_metrics(
        dataset=keyword_matching_dataset,
        engine_class=KeyWordMatchingEngine,
        nb_folds=2,
        intents_filter=["intent2", "intent3"],
        include_slot_metrics=False,
        seed=42,
    )

    expected_metrics = {
        "null": {
            "intent": {
                "true_positive": 0,
                "false_positive": 2,
                "false_negative": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "exact_parsings": 0,
            "intent_utterances": 0,
        },
        "intent2": {
            "intent": {
                "true_positive": 3,
                "false_positive": 0,
                "false_negative": 1,
                "precision": 1.0,
                "recall": 3.0 / 4.0,
                "f1": 0.8571428571428571,
            },
            "exact_parsings": 3,
            "intent_utterances": 4,
        },
        "intent3": {
            "intent": {
                "true_positive": 2,
                "false_positive": 0,
                "false_negative": 1,
                "precision": 1.0,
                "recall": 2.0 / 3.0,
                "f1": 0.8,
            },
            "exact_parsings": 2,
            "intent_utterances": 3,
        },
    }

    assert expected_metrics, res["metrics"]


def test_compute_cross_val_metrics_with_multiple_workers(logger, beverage_dataset):
    # When/Then
    expected_metrics = {
        "null": {
            "intent": {
                "true_positive": 0,
                "false_positive": 11,
                "false_negative": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "exact_parsings": 0,
            "slots": {},
            "intent_utterances": 0,
        },
        "MakeCoffee": {
            "intent": {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 7,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "exact_parsings": 0,
            "slots": {
                "number_of_cups": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
            },
            "intent_utterances": 7,
        },
        "MakeTea": {
            "intent": {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 4,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "exact_parsings": 0,
            "slots": {
                "number_of_cups": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                },
                "beverage_temperature": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                },
            },
            "intent_utterances": 4,
        },
    }
    try:
        res = compute_cross_val_metrics(
            dataset=beverage_dataset, engine_class=MockEngine, nb_folds=2, num_workers=4
        )
    except Exception as e:
        raise AssertionError(e.args[0])
    assert expected_metrics, res["metrics"]


def test_should_raise_when_non_zero_exit(logger, beverage_dataset):
    # When/Then
    with pytest.raises(SystemExit):
        compute_cross_val_metrics(
            dataset=beverage_dataset,
            engine_class=MockEngineSegfault,
            nb_folds=4,
            num_workers=4,
        )


def test_compute_cross_val_metrics_without_slot_metrics(logger, beverage_dataset):
    # When/Then
    try:
        res = compute_cross_val_metrics(
            dataset=beverage_dataset,
            engine_class=MockEngine,
            nb_folds=2,
            include_slot_metrics=False,
        )
    except Exception as e:
        raise AssertionError(e.args[0])

    expected_metrics = {
        "null": {
            "intent": {
                "true_positive": 0,
                "false_positive": 11,
                "false_negative": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "intent_utterances": 0,
            "exact_parsings": 0,
        },
        "MakeCoffee": {
            "intent": {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 7,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "intent_utterances": 7,
            "exact_parsings": 0,
        },
        "MakeTea": {
            "intent": {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 4,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "intent_utterances": 4,
            "exact_parsings": 0,
        },
    }

    assert expected_metrics, res["metrics"]


def test_cross_val_metrics_should_skip_when_not_enough_data(
    logger, beverage_dataset_path
):
    # When
    result = compute_cross_val_metrics(
        dataset=beverage_dataset_path, engine_class=MockEngine, nb_folds=11
    )

    # Then
    expected_result = {
        AVERAGE_METRICS: None,
        CONFUSION_MATRIX: None,
        METRICS: None,
        PARSING_ERRORS: [],
    }
    assert expected_result, result


def test_compute_train_test_metrics(logger, beverage_dataset):
    # When/Then
    try:
        res = compute_train_test_metrics(
            train_dataset=beverage_dataset,
            test_dataset=beverage_dataset,
            engine_class=MockEngine,
        )
    except Exception as e:
        raise AssertionError(e.args[0])

    expected_metrics = {
        "MakeCoffee": {
            "intent": {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 7,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "slots": {
                "number_of_cups": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
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
                "f1": 0.0,
            },
            "slots": {},
            "intent_utterances": 0,
            "exact_parsings": 0,
        },
        "MakeTea": {
            "intent": {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 4,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "slots": {
                "number_of_cups": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                },
                "beverage_temperature": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                },
            },
            "intent_utterances": 4,
            "exact_parsings": 0,
        },
    }

    assert expected_metrics, res["metrics"]


def test_compute_train_test_metrics_without_slots_metrics(logger, beverage_dataset):
    # When/Then
    try:
        res = compute_train_test_metrics(
            train_dataset=beverage_dataset,
            test_dataset=beverage_dataset,
            engine_class=MockEngine,
            include_slot_metrics=False,
        )
    except Exception as e:
        raise AssertionError(e.args[0])

    expected_metrics = {
        "MakeCoffee": {
            "intent": {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 7,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
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
                "f1": 0.0,
            },
            "intent_utterances": 0,
            "exact_parsings": 0,
        },
        "MakeTea": {
            "intent": {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 4,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "intent_utterances": 4,
            "exact_parsings": 0,
        },
    }

    assert expected_metrics, res["metrics"]
