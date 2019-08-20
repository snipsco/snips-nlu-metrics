from __future__ import absolute_import, division, unicode_literals

import inspect
import logging
import sys
from copy import deepcopy

import numpy as np
from future.utils import iteritems, itervalues
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state

from snips_nlu_metrics.utils.constants import (
    DATA, ENTITIES, ENTITY, FALSE_NEGATIVE, FALSE_POSITIVE, INTENTS,
    NONE_INTENT_NAME, SLOT_NAME, TEXT, TRUE_POSITIVE, UTTERANCES,
    EXACT_PARSINGS)
from snips_nlu_metrics.utils.dataset_utils import (
    get_utterances_subset, input_string_from_chunks,
    update_entities_with_utterances)
from snips_nlu_metrics.utils.exception import NotEnoughDataError

logger = logging.getLogger(__name__)

INITIAL_METRICS = {
    TRUE_POSITIVE: 0,
    FALSE_POSITIVE: 0,
    FALSE_NEGATIVE: 0
}


def create_shuffle_stratified_splits(
        dataset, n_splits, train_size_ratio=1.0, drop_entities=False,
        seed=None, out_of_domain_utterances=None, intents_filter=None):
    if train_size_ratio > 1.0 or train_size_ratio < 0:
        error_msg = "Invalid value for train size ratio: %s" % train_size_ratio
        logger.exception(error_msg)
        raise ValueError(error_msg)

    nb_utterances = {intent: len(data[UTTERANCES])
                     for intent, data in iteritems(dataset[INTENTS])}
    if any((nb * train_size_ratio < n_splits
            for nb in itervalues(nb_utterances))):
        raise NotEnoughDataError(dataset, n_splits, train_size_ratio)

    if drop_entities:
        dataset = deepcopy(dataset)
        for entity, data in iteritems(dataset[ENTITIES]):
            data[DATA] = []
    else:
        dataset = update_entities_with_utterances(dataset)

    utterances = np.array([
        (intent_name, utterance)
        for intent_name, intent_data in iteritems(dataset[INTENTS])
        for utterance in intent_data[UTTERANCES]
    ])
    intents = np.array([u[0] for u in utterances])
    X = np.zeros(len(intents))
    random_state = check_random_state(seed)
    sss = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    splits = []
    for train_index, test_index in sss.split(X, intents):
        train_utterances = utterances[train_index].tolist()
        train_utterances = get_utterances_subset(train_utterances,
                                                 train_size_ratio)
        test_utterances = utterances[test_index].tolist()
        train_dataset = deepcopy(dataset)
        train_dataset[INTENTS] = dict()
        for intent_name, utterance in train_utterances:
            if intent_name not in train_dataset[INTENTS]:
                train_dataset[INTENTS][intent_name] = {UTTERANCES: []}
            train_dataset[INTENTS][intent_name][UTTERANCES].append(
                deepcopy(utterance))
        splits.append((train_dataset, test_utterances))

    if intents_filter is not None:
        filtered_splits = []
        for train_dataset, test_utterances in splits:
            test_utterances = [(intent_name, utterance)
                               for intent_name, utterance in test_utterances
                               if intent_name in intents_filter]
            filtered_splits.append((train_dataset, test_utterances))
        splits = filtered_splits

    if out_of_domain_utterances is not None:
        additional_test_utterances = [
            [NONE_INTENT_NAME, {DATA: [{TEXT: utterance}]}]
            for utterance in out_of_domain_utterances
        ]
        for split in splits:
            split[1].extend(additional_test_utterances)

    return splits


def compute_split_metrics(engine_class, split, intent_list,
                          include_slot_metrics, slot_matching_lambda,
                          intents_filter):
    """Fit and run engine on a split specified by train_dataset and
        test_utterances"""
    train_dataset, test_utterances = split
    engine = engine_class()
    engine.fit(train_dataset)
    return compute_engine_metrics(
        engine, test_utterances, intent_list, include_slot_metrics,
        slot_matching_lambda, intents_filter)


def compute_engine_metrics(engine, test_utterances, intent_list,
                           include_slot_metrics, slot_matching_lambda=None,
                           intents_filter=None):
    if slot_matching_lambda is None:
        slot_matching_lambda = exact_match
    metrics = dict()
    intent_list = intent_list + [NONE_INTENT_NAME]
    confusion_matrix = dict(
        intents=intent_list,
        matrix=[[0 for _ in range(len(intent_list))]
                for _ in range(len(intent_list))]
    )
    intents_idx = {
        intent_name: idx for idx, intent_name in enumerate(intent_list)
    }

    errors = []
    for actual_intent, utterance in test_utterances:
        actual_slots = [chunk for chunk in utterance[DATA] if
                        SLOT_NAME in chunk]
        input_string = input_string_from_chunks(utterance[DATA])
        if has_filter_param(engine):
            parsing = engine.parse(input_string, intents_filter=intents_filter)
        else:
            if intents_filter:
                logger.warning("The provided NLU engine (%r) does not support "
                               "intents filter through its `parse` API, "
                               "however one has been passed (%s)", engine,
                               intents_filter)
            parsing = engine.parse(input_string)

        if parsing["intent"] is not None:
            predicted_intent = parsing["intent"]["intentName"]
            if predicted_intent is None:
                predicted_intent = NONE_INTENT_NAME
        else:
            # Use a string here to avoid having a None key in the metrics dict
            predicted_intent = NONE_INTENT_NAME

        predicted_slots = [] if parsing["slots"] is None else parsing["slots"]

        i = intents_idx.get(actual_intent)
        j = intents_idx.get(predicted_intent)

        if i is None or j is None:
            continue

        confusion_matrix["matrix"][i][j] += 1

        utterance_metrics = compute_utterance_metrics(
            predicted_intent, predicted_slots, actual_intent, actual_slots,
            include_slot_metrics, slot_matching_lambda)
        for intent in utterance_metrics:
            utterance_metrics[intent][EXACT_PARSINGS] = 0
        if contains_errors(utterance_metrics, include_slot_metrics):
            if not include_slot_metrics:
                parsing.pop("slots")
            errors.append({
                "nlu_output": parsing,
                "expected_output": format_expected_output(
                    actual_intent, utterance, include_slot_metrics)
            })
        else:
            utterance_metrics[actual_intent][EXACT_PARSINGS] = 1
        metrics = aggregate_metrics(metrics, utterance_metrics,
                                    include_slot_metrics)
    return metrics, errors, confusion_matrix


def has_filter_param(engine):
    if sys.version_info[0] == 2:
        parse_args = inspect.getargspec(engine.parse).args
    else:
        parse_args = inspect.signature(engine.parse).parameters
    return "intents_filter" in parse_args


def compute_utterance_metrics(predicted_intent, predicted_slots, actual_intent,
                              actual_slots, include_slot_metrics,
                              slot_matching_lambda):
    # initialize metrics
    intent_names = {predicted_intent, actual_intent}
    slot_names = set(
        [(predicted_intent, s["slotName"]) for s in predicted_slots] +
        [(actual_intent, u[SLOT_NAME]) for u in actual_slots])

    metrics = dict()
    for intent in intent_names:
        metrics[intent] = {"intent": deepcopy(INITIAL_METRICS)}
        if include_slot_metrics:
            metrics[intent]["slots"] = dict()

    if include_slot_metrics:
        for (intent_name, slot_name) in slot_names:
            metrics[intent_name]["slots"][slot_name] = deepcopy(
                INITIAL_METRICS)

    if predicted_intent == actual_intent:
        metrics[predicted_intent]["intent"][TRUE_POSITIVE] += 1
    else:
        metrics[predicted_intent]["intent"][FALSE_POSITIVE] += 1
        metrics[actual_intent]["intent"][FALSE_NEGATIVE] += 1
        return metrics

    if not include_slot_metrics:
        return metrics

    # Check if expected slots have been parsed
    for slot in actual_slots:
        slot_name = slot[SLOT_NAME]
        slot_metrics = metrics[actual_intent]["slots"][slot_name]
        if any(s["slotName"] == slot_name and slot_matching_lambda(slot, s)
               for s in predicted_slots):
            slot_metrics[TRUE_POSITIVE] += 1
        else:
            slot_metrics[FALSE_NEGATIVE] += 1

    # Check if there are unexpected parsed slots
    for slot in predicted_slots:
        slot_name = slot["slotName"]
        slot_metrics = metrics[predicted_intent]["slots"][slot_name]
        if all(s[SLOT_NAME] != slot_name or not slot_matching_lambda(s, slot)
               for s in actual_slots):
            slot_metrics[FALSE_POSITIVE] += 1
    return metrics


def aggregate_metrics(lhs_metrics, rhs_metrics, include_slot_metrics):
    acc_metrics = deepcopy(lhs_metrics)
    for (intent, intent_metrics) in iteritems(rhs_metrics):
        if intent not in acc_metrics:
            acc_metrics[intent] = deepcopy(intent_metrics)
        else:
            acc_metrics[intent]["intent"] = add_count_metrics(
                acc_metrics[intent]["intent"], intent_metrics["intent"])
            acc_metrics[intent][EXACT_PARSINGS] += intent_metrics[
                EXACT_PARSINGS]
            if not include_slot_metrics:
                continue
            acc_slot_metrics = acc_metrics[intent]["slots"]
            for (slot, slot_metrics) in iteritems(intent_metrics["slots"]):
                if slot not in acc_slot_metrics:
                    acc_slot_metrics[slot] = deepcopy(slot_metrics)
                else:
                    acc_slot_metrics[slot] = add_count_metrics(
                        acc_slot_metrics[slot], slot_metrics)
    return acc_metrics


def aggregate_matrices(lhs_matrix, rhs_matrix):
    if lhs_matrix is None:
        return rhs_matrix
    if rhs_matrix is None:
        return lhs_matrix
    acc_matrix = deepcopy(lhs_matrix)
    matrix_size = len(acc_matrix["matrix"])
    for i in range(matrix_size):
        for j in range(matrix_size):
            acc_matrix["matrix"][i][j] += rhs_matrix["matrix"][i][j]
    return acc_matrix


def add_count_metrics(lhs, rhs):
    return {
        TRUE_POSITIVE: lhs[TRUE_POSITIVE] + rhs[TRUE_POSITIVE],
        FALSE_POSITIVE: lhs[FALSE_POSITIVE] + rhs[FALSE_POSITIVE],
        FALSE_NEGATIVE: lhs[FALSE_NEGATIVE] + rhs[FALSE_NEGATIVE]
    }


def compute_average_metrics(metrics, ignore_none_intent=True):
    metrics = deepcopy(metrics)
    if ignore_none_intent:
        metrics = {
            intent: intent_metrics for intent, intent_metrics in
            iteritems(metrics) if intent and intent != NONE_INTENT_NAME
        }

    nb_intents = len(metrics)
    if not nb_intents:
        return None

    average_intent_f1 = sum(
        intent_metrics["intent"]["f1"]
        for intent, intent_metrics in iteritems(metrics)) / nb_intents
    average_intent_precision = sum(
        intent_metrics["intent"]["precision"]
        for intent, intent_metrics in iteritems(metrics)) / nb_intents
    average_intent_recall = sum(
        intent_metrics["intent"]["recall"]
        for intent, intent_metrics in iteritems(metrics)) / nb_intents

    average_metrics = {
        "intent": {
            "f1": average_intent_f1,
            "precision": average_intent_precision,
            "recall": average_intent_recall,
        },
    }

    nb_slots = sum(1 for intent_metrics in itervalues(metrics)
                   for _ in itervalues(intent_metrics.get("slots", dict())))
    if nb_slots == 0:
        return average_metrics

    average_slot_f1 = sum(
        slot_metrics["f1"]
        for intent_metrics in itervalues(metrics)
        for slot_metrics in itervalues(intent_metrics["slots"])) / nb_slots
    average_slot_precision = sum(
        slot_metrics["precision"]
        for intent_metrics in itervalues(metrics)
        for slot_metrics in itervalues(intent_metrics["slots"])) / nb_slots
    average_slot_recall = sum(
        slot_metrics["recall"]
        for intent_metrics in itervalues(metrics)
        for slot_metrics in itervalues(intent_metrics["slots"])) / nb_slots

    average_metrics["slot"] = {
        "f1": average_slot_f1,
        "precision": average_slot_precision,
        "recall": average_slot_recall,
    }
    return average_metrics


def compute_precision_recall_f1(metrics):
    for intent_metrics in itervalues(metrics):
        prec_rec_metrics = _compute_precision_recall_f1(
            intent_metrics["intent"])
        intent_metrics["intent"].update(prec_rec_metrics)
        if "slots" in intent_metrics:
            for slot_metrics in itervalues(intent_metrics["slots"]):
                prec_rec_metrics = _compute_precision_recall_f1(slot_metrics)
                slot_metrics.update(prec_rec_metrics)
    return metrics


def _compute_precision_recall_f1(count_metrics):
    tp = count_metrics[TRUE_POSITIVE]
    fp = count_metrics[FALSE_POSITIVE]
    fn = count_metrics[FALSE_NEGATIVE]
    precision = 0. if tp == 0 else float(tp) / float(tp + fp)
    recall = 0. if tp == 0 else float(tp) / float(tp + fn)
    if precision == 0. or recall == 0.:
        f1 = 0.
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def contains_errors(utterance_metrics, check_slots):
    for metrics in itervalues(utterance_metrics):
        intent_metrics = metrics["intent"]
        if intent_metrics.get(FALSE_POSITIVE, 0) > 0:
            return True
        if intent_metrics.get(FALSE_NEGATIVE, 0) > 0:
            return True
        if not check_slots:
            continue
        for slot_metrics in itervalues(metrics["slots"]):
            if slot_metrics.get(FALSE_POSITIVE, 0) > 0:
                return True
            if slot_metrics.get(FALSE_NEGATIVE, 0) > 0:
                return True
    return False


def format_expected_output(intent_name, utterance, include_slots):
    char_index = 0
    ranges = []
    for chunk in utterance[DATA]:
        range_end = char_index + len(chunk[TEXT])
        ranges.append({"start": char_index, "end": range_end})
        char_index = range_end

    expected_output = {
        "input": "".join(chunk[TEXT] for chunk in utterance[DATA]),
        "intent": {
            "intentName": intent_name,
            "probability": 1.0
        }
    }
    if include_slots:
        expected_output["slots"] = [
            {
                "rawValue": chunk[TEXT],
                "entity": chunk[ENTITY],
                "slotName": chunk[SLOT_NAME],
                "range": ranges[chunk_index]
            }
            for chunk_index, chunk in enumerate(utterance[DATA])
            if ENTITY in chunk
        ]
    return expected_output


def exact_match(lhs_slot, rhs_slot):
    return lhs_slot[TEXT] == rhs_slot["rawValue"]
