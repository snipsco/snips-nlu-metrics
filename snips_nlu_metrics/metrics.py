from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json

from past.builtins import basestring

from snips_nlu_metrics.engine import Engine, build_nlu_engine_class
from snips_nlu_metrics.utils.constants import (
    INTENTS, UTTERANCES, INTENT_UTTERANCES, PARSING_ERRORS, METRICS)
from snips_nlu_metrics.utils.exception import NotEnoughDataError
from snips_nlu_metrics.utils.metrics_utils import (
    create_shuffle_stratified_splits, compute_engine_metrics,
    aggregate_metrics, compute_precision_recall)


def compute_cross_val_nlu_metrics(dataset, training_engine_class,
                                  inference_engine_class, nb_folds=5,
                                  train_size_ratio=1.0,
                                  drop_entities=False,
                                  slot_matching_lambda=None,
                                  progression_handler=None):
    """Compute pure NLU metrics on the dataset using cross validation

    Args:
        dataset (dict or str): Dataset or path to dataset
        training_engine_class: Python class to use for training
        inference_engine_class: Python class to use for inference
        nb_folds (int, optional): Number of folds to use for cross validation
        train_size_ratio (float, optional): Ratio of intent utterances to use
            for training
        drop_entities (bool, false): Specify whether not all entity values
            should be removed from training data
        slot_matching_lambda (lambda, optional):
            lambda expected_slot, actual_slot -> bool,
            if defined, this function will be use to match slots when computing
            metrics, otherwise exact match will be used.
            `expected_slot` corresponds to the slot as defined in the dataset,
            and `actual_slot` corresponds to the slot as returned by the NLU
        progression_handler (lambda, optional): handler called at each
            progression (%) step

    Returns
        dict: Metrics results containing the following data

            - "metrics": the computed metrics
            - "parsing_errors": the list of parsing errors

    """
    engine_class = build_nlu_engine_class(training_engine_class,
                                          inference_engine_class)
    return compute_cross_val_metrics(dataset, engine_class, nb_folds,
                                     train_size_ratio, drop_entities,
                                     slot_matching_lambda, progression_handler)


def compute_cross_val_metrics(dataset, engine_class, nb_folds=5,
                              train_size_ratio=1.0, drop_entities=False,
                              slot_matching_lambda=None,
                              progression_handler=None):
    """Compute end-to-end metrics on the dataset using cross validation

    Args:
        dataset (dict or str): Dataset or path to dataset
        engine_class: Python class to use for training and inference, this
            class must inherit from `Engine`
        nb_folds (int, optional): Number of folds to use for cross validation
        train_size_ratio: float, ratio of intent utterances to use for
            training (default=5)
        drop_entities (bool, false): Specify whether not all entity values
            should be removed from training data
        slot_matching_lambda (lambda, optional):
            lambda expected_slot, actual_slot -> bool,
            if defined, this function will be use to match slots when computing
            metrics, otherwise exact match will be used.
            `expected_slot` corresponds to the slot as defined in the dataset,
            and `actual_slot` corresponds to the slot as returned by the NLU
        progression_handler (lambda, optional): handler called at each
            progression (%) step

    Returns:
        dict: Metrics results containing the following data

            - "metrics": the computed metrics
            - "parsing_errors": the list of parsing errors

    """
    if not issubclass(engine_class, Engine):
        print("WARNING: %s does not inherit from %s" % (engine_class, Engine))

    if isinstance(dataset, basestring):
        with io.open(dataset, encoding="utf8") as f:
            dataset = json.load(f)

    try:
        splits = create_shuffle_stratified_splits(
            dataset, nb_folds, train_size_ratio, drop_entities)
    except NotEnoughDataError as e:
        print("Skipping metrics computation because of: %s" % e.message)
        return {
            METRICS: None,
            PARSING_ERRORS: []
        }
    global_metrics = dict()

    global_errors = []
    total_splits = len(splits)
    for split_index, (train_dataset, test_utterances) in enumerate(splits):
        engine = engine_class()
        engine.fit(train_dataset)
        split_metrics, errors = compute_engine_metrics(engine, test_utterances,
                                                       slot_matching_lambda)
        global_metrics = aggregate_metrics(global_metrics, split_metrics)
        global_errors += errors
        if progression_handler is not None:
            progression_handler(float(split_index + 1) / float(total_splits))

    global_metrics = compute_precision_recall(global_metrics)

    nb_utterances = {intent: len(data[UTTERANCES])
                     for intent, data in dataset[INTENTS].items()}
    for intent, metrics in global_metrics.items():
        metrics[INTENT_UTTERANCES] = nb_utterances.get(intent, 0)

    return {
        METRICS: global_metrics,
        PARSING_ERRORS: global_errors
    }


def compute_train_test_nlu_metrics(train_dataset, test_dataset,
                                   training_engine_class,
                                   inference_engine_class,
                                   slot_matching_lambda=None):
    """Compute pure NLU metrics on `test_dataset` after having trained on
    `train_dataset`

    Args
        train_dataset (dict or str): Dataset or path to dataset used for
            training
        test_dataset (dict or str): Dataset or path to dataset used for testing
        training_engine_class: Python class to use for training
        inference_engine_class: Python class to use for inference
        slot_matching_lambda (lambda, optional):
            lambda expected_slot, actual_slot -> bool,
            if defined, this function will be use to match slots when computing
            metrics, otherwise exact match will be used.
            `expected_slot` corresponds to the slot as defined in the dataset,
            and `actual_slot` corresponds to the slot as returned by the NLU

    Returns
        dict: Metrics results containing the following data

            - "metrics": the computed metrics
            - "parsing_errors": the list of parsing errors
    """
    engine_class = build_nlu_engine_class(training_engine_class,
                                          inference_engine_class)
    return compute_train_test_metrics(train_dataset, test_dataset,
                                      engine_class, slot_matching_lambda)


def compute_train_test_metrics(train_dataset, test_dataset, engine_class,
                               slot_matching_lambda=None):
    """Compute end-to-end metrics on `test_dataset` after having trained on
    `train_dataset`

    Args:
        train_dataset (dict or str): Dataset or path to dataset used for
            training
        test_dataset (dict or str): dataset or path to dataset used for testing
        engine_class: Python class to use for training and inference, this
            class must inherit from `Engine`
        slot_matching_lambda (lambda, optional):
            lambda expected_slot, actual_slot -> bool,
            if defined, this function will be use to match slots when computing
            metrics, otherwise exact match will be used.
            `expected_slot` corresponds to the slot as defined in the dataset,
            and `actual_slot` corresponds to the slot as returned by the NLU

    Returns
        dict: Metrics results containing the following data

            - "metrics": the computed metrics
            - "parsing_errors": the list of parsing errors
    """
    if not issubclass(engine_class, Engine):
        print("WARNING: %s does not inherit from %s" % (engine_class, Engine))

    if isinstance(train_dataset, basestring):
        with io.open(train_dataset, encoding="utf8") as f:
            train_dataset = json.load(f)

    if isinstance(test_dataset, basestring):
        with io.open(test_dataset, encoding="utf8") as f:
            test_dataset = json.load(f)

    engine = engine_class()
    engine.fit(train_dataset)
    test_utterances = [
        (intent_name, utterance)
        for intent_name, intent_data in test_dataset[INTENTS].items()
        for utterance in intent_data[UTTERANCES]
    ]
    metrics, errors = compute_engine_metrics(engine, test_utterances,
                                             slot_matching_lambda)
    metrics = compute_precision_recall(metrics)
    nb_utterances = {intent: len(data[UTTERANCES])
                     for intent, data in train_dataset[INTENTS].items()}
    for intent, intent_metrics in metrics.items():
        intent_metrics[INTENT_UTTERANCES] = nb_utterances.get(intent, 0)
    return {
        METRICS: metrics,
        PARSING_ERRORS: errors
    }
