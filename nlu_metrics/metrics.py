from __future__ import unicode_literals

import io
import json

from nlu_metrics.utils.dataset_utils import get_stratified_utterances
from nlu_metrics.utils.exception import NotEnoughDataError
from nlu_metrics.utils.metrics_utils import (create_k_fold_batches,
                                             compute_engine_metrics,
                                             aggregate_metrics,
                                             compute_precision_recall,
                                             exact_match)
from nlu_metrics.utils.nlu_engine_utils import (get_inference_engine,
                                                get_trained_engine)
from nlu_metrics.utils.constants import INTENTS, UTTERANCES


def compute_cross_val_metrics(
        dataset,
        training_engine_class,
        inference_engine_class,
        nb_folds=5,
        train_size_ratio=1.0,
        use_asr_output=False,
        slot_matching_lambda=None):
    """Compute the main NLU metrics on the dataset using cross validation

    :param dataset: dict or str, dataset or path to dataset
    :param training_engine_class: python class to use for training
    :param inference_engine_class: python class to use for inference
    :param nb_folds: int, number of folds to use for cross validation
    :param train_size_ratio: float, ratio of intent utterances to use for
        training
    :param use_asr_output: bool (optional), whether the asr output should be
        used instead of utterance text
    :param slot_matching_lambda: lambda lhs_slot, rhs_slot: bool (optional),
        if defined, this function will be use to match slots when computing
        metrics, otherwise exact match will be used
    :return: dict containing the metrics

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
        trained_engine = get_trained_engine(train_dataset,
                                            training_engine_class)
        inference_engine = get_inference_engine(language,
                                                trained_engine.to_dict(),
                                                inference_engine_class)
        batch_metrics, errors = compute_engine_metrics(
            inference_engine, test_utterances, use_asr_output,
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


def compute_train_test_metrics(
        train_dataset,
        test_dataset,
        training_engine_class,
        inference_engine_class,
        use_asr_output=False,
        slot_matching_lambda=None):
    """Compute the main NLU metrics on `test_dataset` after having trained on
    `train_dataset`

    :param train_dataset: dict or str, dataset or path to dataset used for
        training
    :param test_dataset: dict or str, dataset or path to dataset used for
        testing
    :param training_engine_class: python class to use for training
    :param inference_engine_class: python class to use for inference
    :param use_asr_output: bool (optional), whether the asr output should be
        used instead of utterance text
    :param slot_matching_lambda: lambda lhs_slot, rhs_slot: bool (optional),
        if defined, this function will be use to match slots when computing
        metrics, otherwise exact match will be used
    :return: dict containing the metrics
    """
    if isinstance(train_dataset, (str, unicode)):
        with io.open(train_dataset, encoding="utf8") as f:
            train_dataset = json.load(f)

    if isinstance(test_dataset, (str, unicode)):
        with io.open(test_dataset, encoding="utf8") as f:
            test_dataset = json.load(f)

    if slot_matching_lambda is None:
        slot_matching_lambda = exact_match

    language = train_dataset["language"]
    trained_engine = get_trained_engine(train_dataset, training_engine_class)
    inference_engine = get_inference_engine(language, trained_engine.to_dict(),
                                            inference_engine_class)
    utterances = get_stratified_utterances(test_dataset, seed=None,
                                           shuffle=False)
    metrics, errors = compute_engine_metrics(
        inference_engine, utterances, use_asr_output, slot_matching_lambda)
    metrics = compute_precision_recall(metrics)
    return {"metrics": metrics, "errors": errors}
