from __future__ import division, print_function, unicode_literals

import io
import json
import logging
from builtins import map

from future.utils import iteritems
from joblib import Parallel, delayed
from past.builtins import basestring

from snips_nlu_metrics.utils.constants import (
    AVERAGE_METRICS, CONFUSION_MATRIX, INTENTS, INTENT_UTTERANCES, METRICS,
    PARSING_ERRORS, EXACT_PARSINGS, UTTERANCES)
from snips_nlu_metrics.utils.exception import NotEnoughDataError
from snips_nlu_metrics.utils.metrics_utils import (
    aggregate_matrices, aggregate_metrics, compute_average_metrics,
    compute_engine_metrics, compute_precision_recall_f1, compute_split_metrics,
    create_shuffle_stratified_splits)

logger = logging.getLogger(__name__)


# Ensure back-compatibility
def compute_cross_val_metrics(*args, **kwargs):
    if 'list_runtime_params' in kwargs:
        raise RuntimeError(
            "list_runtime_params must not be set when using "
            "this version of compute_cross_val_metrics.")
    return compute_cross_val_metrics_grid_runtime(*args, **kwargs)[0]


def compute_cross_val_metrics_grid_runtime(
        dataset, engine_class, nb_folds=5, train_size_ratio=1.0,
        drop_entities=False, include_slot_metrics=True,
        slot_matching_lambda=None, progression_handler=None, num_workers=1,
        seed=None, out_of_domain_utterances=None,
        include_exact_parsings=False, list_runtime_params=None):
    """Compute end-to-end metrics on the dataset using cross validation

    Args:
        dataset (dict or str): Dataset or path to dataset
        engine_class: Python class to use for training and inference, this
            class must inherit from `Engine`
        nb_folds (int, optional): Number of folds to use for cross validation
            (default=5)
        train_size_ratio (float, optional): ratio of intent utterances to use
            for training (default=1.0)
        drop_entities (bool, optional): Specify whether or not all entity
            values should be removed from training data (default=False)
        include_slot_metrics (bool, optional): If false, the slots metrics and
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
        out_of_domain_utterances (list, optional): If defined, list of 
            out-of-domain utterances to be added to the pool of test utterances 
            in each split
        include_exact_parsings (bool, optional): If true, include exact 
            parsings in persisted parsings

    Returns:
        dict: Metrics results containing the following data
    
            - "metrics": the computed metrics
            - "parsing_errors": the list of parsing errors
            - "exact_parsings": the list of exact parsings
            - "confusion_matrix": the computed confusion matrix
            - "average_metrics": the metrics averaged over all intents    
    """

    if list_runtime_params is None:
        # Only default runtime parameters
        list_runtime_params = [None]

    if isinstance(dataset, basestring):
        with io.open(dataset, encoding="utf8") as f:
            dataset = json.load(f)

    try:
        splits = create_shuffle_stratified_splits(
            dataset, nb_folds, train_size_ratio, drop_entities,
            seed, out_of_domain_utterances)
    except NotEnoughDataError as e:
        logger.warning("Skipping metrics computation because of: %s"
                       % e.message)
        return [{
            AVERAGE_METRICS: None,
            CONFUSION_MATRIX: None,
            METRICS: None,
            PARSING_ERRORS: [],
        }]

    intent_list = sorted(list(dataset["intents"]))

    list_global_metrics = [dict() for _ in range(len(list_runtime_params))]
    list_global_confusion_matrix = [None for _ in range(len(list_runtime_params))]
    list_global_errors = [[] for _ in range(len(list_runtime_params))]
    list_global_exact_parsings = [[] for _ in range(len(list_runtime_params))]

    total_splits = len(splits)

    def compute_metrics(split_):
        logger.info("Computing metrics for dataset split ...")
        return compute_split_metrics(
            engine_class, split_, intent_list, include_slot_metrics,
            slot_matching_lambda, include_exact_parsings,
            list_runtime_params=list_runtime_params)

    effective_num_workers = min(num_workers, len(splits))
    if effective_num_workers > 1:
        parallel = Parallel(n_jobs=effective_num_workers)
        results = parallel(delayed(compute_metrics)(split) for split in splits)
    else:
        results = map(compute_metrics, splits)

    for split_index, list_results in enumerate(results):
        for runtime_params_idx in range(len(list_runtime_params)):
            split_metrics, errors, exact_parsings, confusion_matrix = list_results[runtime_params_idx]

            list_global_metrics[runtime_params_idx] = aggregate_metrics(
                list_global_metrics[runtime_params_idx], split_metrics, include_slot_metrics)
            list_global_confusion_matrix[runtime_params_idx] = aggregate_matrices(
                list_global_confusion_matrix[runtime_params_idx], confusion_matrix)
            list_global_errors[runtime_params_idx] += errors
            if include_exact_parsings:
                list_global_exact_parsings[runtime_params_idx] += exact_parsings
        logger.info("Done computing %d/%d splits"
                % (split_index + 1, total_splits))

        if progression_handler is not None:
            progression_handler(
                float(split_index + 1) / float(total_splits))

    for runtime_params_idx in range(len(list_runtime_params)):
        list_global_metrics[runtime_params_idx] = compute_precision_recall_f1(
            list_global_metrics[runtime_params_idx])

    list_average_metrics = [compute_average_metrics(
        list_global_metrics[runtime_params_idx],
        ignore_none_intent=True if out_of_domain_utterances is None else False)
        for runtime_params_idx in range(len(list_runtime_params))
    ]

    nb_utterances = {intent: len(data[UTTERANCES])
                     for intent, data in iteritems(dataset[INTENTS])}

    for runtime_params_idx in range(len(list_runtime_params)):
        for intent, metrics in iteritems(list_global_metrics[runtime_params_idx]):
            metrics[INTENT_UTTERANCES] = nb_utterances.get(intent, 0)

    return [{
        CONFUSION_MATRIX: list_global_confusion_matrix[runtime_params_idx],
        AVERAGE_METRICS: list_average_metrics[runtime_params_idx],
        METRICS: list_global_metrics[runtime_params_idx],
        PARSING_ERRORS: list_global_errors[runtime_params_idx],
        EXACT_PARSINGS: list_global_exact_parsings[runtime_params_idx],
    } for runtime_params_idx in range(len(list_runtime_params))]


# Ensure back-compatibilty
def compute_train_test_metrics(*args, **kwargs):
    if 'list_runtime_params' in kwargs:
        raise RuntimeError(
            "list_runtime_params must not be set when using "
            "this version of compute_cross_val_metrics.")
    return compute_train_test_metrics_grid_runtime(*args, **kwargs)[0]


def compute_train_test_metrics_grid_runtime(
        train_dataset, test_dataset, engine_class, include_slot_metrics=True,
        slot_matching_lambda=None, include_exact_parsings=False,
        list_runtime_params=None):
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
        include_exact_parsings (bool, optional): If true, include exact 
            parsings in persisted parsings

    Returns
        dict: Metrics results containing the following data

            - "metrics": the computed metrics
            - "parsing_errors": the list of parsing errors
            - "exact_parsings": the list of exact parsings
            - "confusion_matrix": the computed confusion matrix
            - "average_metrics": the metrics averaged over all intents
    """

    if list_runtime_params is None:
        # Only default runtime parameters
        list_runtime_params = [None]

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
    ]
    results = []
    logger.info("Computing metrics...")
    for runtime_params in list_runtime_params:
        metrics, errors, exact_parsings, confusion_matrix = compute_engine_metrics(
            engine, test_utterances, intent_list, include_slot_metrics,
            slot_matching_lambda, include_exact_parsings,
            runtime_params=runtime_params)
        metrics = compute_precision_recall_f1(metrics)
        average_metrics = compute_average_metrics(metrics)
        nb_utterances = {intent: len(data[UTTERANCES])
                         for intent, data in iteritems(train_dataset[INTENTS])}
        for intent, intent_metrics in iteritems(metrics):
            intent_metrics[INTENT_UTTERANCES] = nb_utterances.get(intent, 0)
        results.append({
            CONFUSION_MATRIX: confusion_matrix,
            AVERAGE_METRICS: average_metrics,
            METRICS: metrics,
            PARSING_ERRORS: errors,
            EXACT_PARSINGS: exact_parsings
        })
    return results

