from __future__ import division, print_function, unicode_literals

import io
import json

from past.builtins import basestring
from pathos.multiprocessing import Pool

from snips_nlu_metrics.utils.constants import (
    CONFUSION_MATRIX, INTENTS, INTENT_UTTERANCES, METRICS, PARSING_ERRORS,
    UTTERANCES)
from snips_nlu_metrics.utils.exception import NotEnoughDataError
from snips_nlu_metrics.utils.metrics_utils import (
    aggregate_matrices, aggregate_metrics, compute_engine_metrics,
    compute_precision_recall_f1, create_shuffle_stratified_splits)


def compute_cross_val_metrics(dataset, engine_class, nb_folds=5,
                              train_size_ratio=1.0, drop_entities=False,
                              include_slot_metrics=True,
                              slot_matching_lambda=None,
                              progression_handler=None,
                              num_workers=1):
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
        include_slot_metrics (bool, true): If false, the slots metrics and the
            slots parsing errors will not be reported.
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

    intent_list = sorted(list(dataset["intents"]))
    global_metrics = dict()
    global_confusion_matrix = None
    global_errors = []
    total_splits = len(splits)

    def run_split(engine_class, train_dataset, test_utterances):
        """
            Fit and run engine on a split specified by train_dataset and
            test_utterances
        """
        engine = engine_class()
        engine.fit(train_dataset)
        return compute_engine_metrics(
            engine, test_utterances, intent_list, include_slot_metrics,
            slot_matching_lambda)

    def update_metrics(global_metrics,
                       split_metrics,
                       global_confusion_matrix,
                       confusion_matrix,
                       global_errors,
                       errors):
        """
            Update global metrics with results on split
        """
        global_metrics = aggregate_metrics(global_metrics,
                                           split_metrics,
                                           include_slot_metrics)
        global_confusion_matrix = \
            aggregate_matrices(global_confusion_matrix,
                               confusion_matrix)
        global_errors += errors
        return global_metrics, global_confusion_matrix, global_errors

    if num_workers > 1:
        effective_num_workers = min(num_workers, len(splits))
        pool = Pool(effective_num_workers)
        results = pool.map(
            lambda (train_dataset, test_utterances):
            run_split(engine_class,
                      train_dataset,
                      test_utterances),
            splits)
        for split_metrics, errors, confusion_matrix in results:
            global_metrics, global_confusion_matrix, global_errors = \
                update_metrics(global_metrics,
                               split_metrics,
                               global_confusion_matrix,
                               confusion_matrix,
                               global_errors, errors)
    else:
        for split_index, (train_dataset, test_utterances) in enumerate(splits):
            split_metrics, errors, confusion_matrix = run_split(
                engine_class,
                train_dataset,
                test_utterances)
            global_metrics, global_confusion_matrix, global_errors = \
                update_metrics(global_metrics,
                               split_metrics,
                               global_confusion_matrix,
                               confusion_matrix,
                               global_errors, errors)
            if progression_handler is not None:
                progression_handler(
                    float(split_index + 1) / float(total_splits))

    global_metrics = compute_precision_recall_f1(global_metrics)

    nb_utterances = {intent: len(data[UTTERANCES])
                     for intent, data in dataset[INTENTS].items()}
    for intent, metrics in global_metrics.items():
        metrics[INTENT_UTTERANCES] = nb_utterances.get(intent, 0)

    return {
        METRICS: global_metrics,
        PARSING_ERRORS: global_errors,
        CONFUSION_MATRIX: global_confusion_matrix
    }


def compute_train_test_metrics(train_dataset, test_dataset, engine_class,
                               include_slot_metrics=True,
                               slot_matching_lambda=None):
    """Compute end-to-end metrics on `test_dataset` after having trained on
    `train_dataset`

    Args:
        train_dataset (dict or str): Dataset or path to dataset used for
            training
        test_dataset (dict or str): dataset or path to dataset used for testing
        engine_class: Python class to use for training and inference, this
            class must inherit from `Engine`
        include_slot_metrics (bool, true): If false, the slots metrics and the
            slots parsing errors will not be reported.
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

    if isinstance(train_dataset, basestring):
        with io.open(train_dataset, encoding="utf8") as f:
            train_dataset = json.load(f)

    if isinstance(test_dataset, basestring):
        with io.open(test_dataset, encoding="utf8") as f:
            test_dataset = json.load(f)

    intent_list = set(train_dataset["intents"])
    intent_list.update(test_dataset["intents"])
    intent_list = sorted(intent_list)

    engine = engine_class()
    engine.fit(train_dataset)
    test_utterances = [
        (intent_name, utterance)
        for intent_name, intent_data in test_dataset[INTENTS].items()
        for utterance in intent_data[UTTERANCES]
    ]
    metrics, errors, confusion_matrix = compute_engine_metrics(
        engine, test_utterances, intent_list, include_slot_metrics,
        slot_matching_lambda)
    metrics = compute_precision_recall_f1(metrics)
    nb_utterances = {intent: len(data[UTTERANCES])
                     for intent, data in train_dataset[INTENTS].items()}
    for intent, intent_metrics in metrics.items():
        intent_metrics[INTENT_UTTERANCES] = nb_utterances.get(intent, 0)
    return {
        METRICS: metrics,
        PARSING_ERRORS: errors,
        CONFUSION_MATRIX: confusion_matrix
    }
