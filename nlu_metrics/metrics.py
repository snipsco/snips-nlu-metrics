from __future__ import unicode_literals

import io
import json

from nlu_metrics.engine import Engine
from nlu_metrics.utils.constants import INTENTS, UTTERANCES
from nlu_metrics.utils.dataset_utils import get_stratified_utterances
from nlu_metrics.utils.exception import NotEnoughDataError
from nlu_metrics.utils.metrics_utils import (create_k_fold_batches,
                                             compute_engine_metrics,
                                             aggregate_metrics,
                                             compute_precision_recall,
                                             exact_match)


def compute_cross_val_metrics(dataset, engine_class, nb_folds=5,
                              train_size_ratio=1.0, slot_matching_lambda=None):
    """Compute the main NLU metrics on the dataset using cross validation

    :param dataset: dict or str, dataset or path to dataset
    :param engine_class: python class to use for training and inference
    :param nb_folds: int, number of folds to use for cross validation
    :param train_size_ratio: float, ratio of intent utterances to use for
        training
    :param slot_matching_lambda: lambda lhs_slot, rhs_slot: bool (optional),
        if defined, this function will be use to match slots when computing
        metrics, otherwise exact match will be used
    :return: dict containing the following data

        - "config": the config use to compute the metrics
        - "metrics": the computed metrics
        - "errors": the list of parsing errors

    """

    assert 0.0 <= train_size_ratio <= 1.0

    if slot_matching_lambda is None:
        slot_matching_lambda = exact_match

    metrics_config = {
        "nb_folds": nb_folds,
        "train_size_ratio": train_size_ratio
    }

    if isinstance(dataset, (str, unicode)):
        with io.open(dataset, encoding="utf8") as f:
            dataset = json.load(f)

    nb_utterances = {intent: len(data[UTTERANCES])
                     for intent, data in dataset[INTENTS].iteritems()}
    total_utterances = sum(nb_utterances.values())
    if total_utterances < nb_folds:
        message = "number of utterances is too low (%s)" % total_utterances
        print("Skipping group because of: %s" % message)
        return {
            "config": metrics_config,
            "training_info": message,
            "metrics": None
        }
    try:
        batches = create_k_fold_batches(dataset, nb_folds, train_size_ratio)
    except NotEnoughDataError as e:
        print("Skipping group because of: %s" % e.message)
        return {
            "config": metrics_config,
            "training_info": e.message,
            "metrics": None
        }
    global_metrics = dict()

    global_errors = []
    for batch_index, (train_dataset, test_utterances) in enumerate(batches):
        language = train_dataset["language"]
        engine = engine_class(language)
        engine.fit(train_dataset)
        batch_metrics, errors = compute_engine_metrics(engine, test_utterances,
                                                       slot_matching_lambda)
        global_metrics = aggregate_metrics(global_metrics, batch_metrics)
        global_errors += errors

    global_metrics = compute_precision_recall(global_metrics)

    for intent, metrics in global_metrics.iteritems():
        metrics["intent_utterances"] = nb_utterances.get(intent, 0)

    return {
        "config": metrics_config,
        "metrics": global_metrics,
        "errors": global_errors
    }


def compute_train_test_metrics(train_dataset, test_dataset, engine_class,
                               slot_matching_lambda=None):
    """Compute the main NLU metrics on `test_dataset` after having trained on
    `train_dataset`

    :param train_dataset: dict or str, dataset or path to dataset used for
        training
    :param test_dataset: dict or str, dataset or path to dataset used for
        testing
    :param engine_class: python class to use for training and inference
    :param slot_matching_lambda: lambda lhs_slot, rhs_slot: bool (optional),
        if defined, this function will be use to match slots when computing
        metrics, otherwise exact match will be used
    :return: dict containing the following data

        - "metrics": the computed metrics
        - "errors": the list of parsing errors
    """
    if not issubclass(engine_class, Engine):
        raise TypeError("%s does not inherit from %s" % (engine_class, Engine))

    if isinstance(train_dataset, (str, unicode)):
        with io.open(train_dataset, encoding="utf8") as f:
            train_dataset = json.load(f)

    if isinstance(test_dataset, (str, unicode)):
        with io.open(test_dataset, encoding="utf8") as f:
            test_dataset = json.load(f)

    if slot_matching_lambda is None:
        slot_matching_lambda = exact_match

    language = train_dataset["language"]
    engine = engine_class(language)
    engine.fit(train_dataset)
    utterances = get_stratified_utterances(test_dataset, seed=None,
                                           shuffle=False)
    metrics, errors = compute_engine_metrics(engine, utterances,
                                             slot_matching_lambda)
    metrics = compute_precision_recall(metrics)
    return {"metrics": metrics, "errors": errors}
