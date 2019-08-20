from __future__ import division, print_function, unicode_literals

import io
import json
import logging

from future.utils import iteritems
from joblib import Parallel, delayed
from past.builtins import basestring

from snips_nlu_metrics.utils.constants import (
    AVERAGE_METRICS, CONFUSION_MATRIX, INTENTS, INTENT_UTTERANCES, METRICS,
    PARSING_ERRORS, UTTERANCES)
from snips_nlu_metrics.utils.exception import NotEnoughDataError
from snips_nlu_metrics.utils.metrics_utils import (
    aggregate_matrices, aggregate_metrics, compute_average_metrics,
    compute_engine_metrics, compute_precision_recall_f1, compute_split_metrics,
    create_shuffle_stratified_splits)

logger = logging.getLogger(__name__)


def compute_cross_val_metrics(
        dataset, engine_class, nb_folds=5, train_size_ratio=1.0,
        drop_entities=False, include_slot_metrics=True,
        slot_matching_lambda=None, progression_handler=None, num_workers=1,
        seed=None, out_of_domain_utterances=None, intents_filter=None):
    """Compute end-to-end metrics on the dataset using cross validation

    Args:
        dataset (dict or str): dataset or path to dataset
        engine_class: python class to use for training and inference, this
            class must inherit from `Engine`
        nb_folds (int, optional): number of folds to use for cross validation
            (default=5)
        train_size_ratio (float, optional): ratio of intent utterances to use
            for training (default=1.0)
        drop_entities (bool, optional): specify whether or not all entity
            values should be removed from training data (default=False)
        include_slot_metrics (bool, optional): if false, the slots metrics and
            the slots parsing errors will not be reported (default=True)
        slot_matching_lambda (lambda, optional):
            lambda expected_slot, actual_slot -> bool,
            if defined, this function will be use to match slots when computing
            metrics, otherwise exact match will be used.
            `expected_slot` corresponds to the slot as defined in the dataset,
            and `actual_slot` corresponds to the slot as returned by the NLU
            default(None)
        progression_handler (lambda, optional): handler called at each
            progression (%) step (default=None)
        num_workers (int, optional): number of workers to use. Each worker
            is assigned a certain number of splits (default=1)
        seed (int, optional): seed for the split creation
        out_of_domain_utterances (list, optional): if defined, list of
            out-of-domain utterances to be added to the pool of test utterances
            in each split
        intents_filter (list of str, optional): if defined, at inference times
            test utterances will be restricted to the ones belonging to this
            filter. Moreover, if the parsing API allows it, the inference will
            be made using this intents filter.

    Returns:
        dict: Metrics results containing the following data
    
            - "metrics": the computed metrics
            - "parsing_errors": the list of parsing errors
            - "confusion_matrix": the computed confusion matrix
            - "average_metrics": the metrics averaged over all intents    
    """

    if isinstance(dataset, basestring):
        with io.open(dataset, encoding="utf8") as f:
            dataset = json.load(f)

    try:
        splits = create_shuffle_stratified_splits(
            dataset, nb_folds, train_size_ratio, drop_entities,
            seed, out_of_domain_utterances, intents_filter)
    except NotEnoughDataError as e:
        logger.warning("Not enough data, skipping metrics computation: %r", e)
        return {
            AVERAGE_METRICS: None,
            CONFUSION_MATRIX: None,
            METRICS: None,
            PARSING_ERRORS: [],
        }

    intent_list = sorted(list(dataset["intents"]))
    global_metrics = dict()
    global_confusion_matrix = None
    global_errors = []
    total_splits = len(splits)

    def compute_metrics(split_):
        logger.info("Computing metrics for dataset split ...")
        return compute_split_metrics(
            engine_class, split_, intent_list, include_slot_metrics,
            slot_matching_lambda, intents_filter)

    effective_num_workers = min(num_workers, len(splits))
    if effective_num_workers > 1:
        parallel = Parallel(n_jobs=effective_num_workers)
        results = parallel(delayed(compute_metrics)(split) for split in splits)
    else:
        results = [compute_metrics(s) for s in splits]

    for result in enumerate(results):
        split_index, (split_metrics, errors, confusion_matrix) = result
        global_metrics = aggregate_metrics(
            global_metrics, split_metrics, include_slot_metrics)
        global_confusion_matrix = aggregate_matrices(
            global_confusion_matrix, confusion_matrix)
        global_errors += errors
        logger.info("Done computing %d/%d splits"
                    % (split_index + 1, total_splits))

        if progression_handler is not None:
            progression_handler(
                float(split_index + 1) / float(total_splits))

    global_metrics = compute_precision_recall_f1(global_metrics)

    average_metrics = compute_average_metrics(
        global_metrics,
        ignore_none_intent=True if out_of_domain_utterances is None else False)

    nb_utterances = {intent: len(data[UTTERANCES])
                     for intent, data in iteritems(dataset[INTENTS])}
    for intent, metrics in iteritems(global_metrics):
        metrics[INTENT_UTTERANCES] = nb_utterances.get(intent, 0)

    return {
        CONFUSION_MATRIX: global_confusion_matrix,
        AVERAGE_METRICS: average_metrics,
        METRICS: global_metrics,
        PARSING_ERRORS: global_errors,
    }


def compute_train_test_metrics(
        train_dataset, test_dataset, engine_class, include_slot_metrics=True,
        slot_matching_lambda=None, intents_filter=None):
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
        intents_filter (list of str, optional): if defined, at inference times
            test utterances will be restricted to the ones belonging to this
            filter. Moreover, if the parsing API allows it, the inference will
            be made using this intents filter.

    Returns
        dict: Metrics results containing the following data

            - "metrics": the computed metrics
            - "parsing_errors": the list of parsing errors
            - "confusion_matrix": the computed confusion matrix
            - "average_metrics": the metrics averaged over all intents
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

    logger.info("Training engine...")
    engine = engine_class()
    engine.fit(train_dataset)
    test_utterances = [
        (intent_name, utterance)
        for intent_name, intent_data in iteritems(test_dataset[INTENTS])
        for utterance in intent_data[UTTERANCES]
        if intents_filter is None or intent_name in intents_filter
    ]

    logger.info("Computing metrics...")
    metrics, errors, confusion_matrix = compute_engine_metrics(
        engine, test_utterances, intent_list, include_slot_metrics,
        slot_matching_lambda, intents_filter)
    metrics = compute_precision_recall_f1(metrics)
    average_metrics = compute_average_metrics(metrics)
    nb_utterances = {intent: len(data[UTTERANCES])
                     for intent, data in iteritems(train_dataset[INTENTS])}
    for intent, intent_metrics in iteritems(metrics):
        intent_metrics[INTENT_UTTERANCES] = nb_utterances.get(intent, 0)
    return {
        CONFUSION_MATRIX: confusion_matrix,
        AVERAGE_METRICS: average_metrics,
        METRICS: metrics,
        PARSING_ERRORS: errors,
    }
