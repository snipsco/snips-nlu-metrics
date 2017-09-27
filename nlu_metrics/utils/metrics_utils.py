from __future__ import unicode_literals

from copy import deepcopy

from constants import (INTENTS, UTTERANCES, DATA, SLOT_NAME, TEXT,
                       FALSE_POSITIVE, FALSE_NEGATIVE, ENTITY,
                       TRUE_POSITIVE)
from nlu_metrics.utils.dataset_utils import (input_string_from_chunks,
                                             get_stratified_utterances,
                                             get_utterances_subset)
from nlu_metrics.utils.exception import NotEnoughDataError

INITIAL_METRICS = {
    TRUE_POSITIVE: 0,
    FALSE_POSITIVE: 0,
    FALSE_NEGATIVE: 0
}

NONE_INTENT_NAME = "null"


def create_k_fold_batches(dataset, k, train_size_ratio=1.0, seed=None):
    dataset = deepcopy(dataset)
    # Remove entity values in order to un-bias the cross validation
    for name, entity in dataset["entities"].iteritems():
        if is_builtin_entity(name):
            continue
        if entity["automatically_extensible"]:
            entity["data"] = []

    utterances = get_stratified_utterances(dataset, seed, shuffle=True)
    k_fold_batches = []
    batch_size = len(utterances) / k
    for batch_index in xrange(k):
        test_start = batch_index * batch_size
        test_end = (batch_index + 1) * batch_size
        train_utterances = utterances[0:test_start] + utterances[test_end:]
        train_utterances = get_utterances_subset(train_utterances,
                                                 train_size_ratio)
        if len(train_utterances) == 0:
            raise NotEnoughDataError("Not enough data given the other "
                                     "parameters "
                                     "(nb_folds=%s, train_size_ratio=%s)"
                                     % (k, train_size_ratio))
        test_utterances = utterances[test_start: test_end]
        train_dataset = deepcopy(dataset)
        train_dataset[INTENTS] = dict()
        for intent_name, utterance in train_utterances:
            if intent_name not in train_dataset[INTENTS]:
                train_dataset[INTENTS][intent_name] = {
                    UTTERANCES: [],
                    "engineType": "regex"
                }
            train_dataset[INTENTS][intent_name][UTTERANCES].append(
                deepcopy(utterance))
        k_fold_batches.append((train_dataset, test_utterances))
    return k_fold_batches


def compute_engine_metrics(engine, test_utterances, slot_matching_lambda):
    metrics = dict()
    errors = []
    for intent_name, utterance in test_utterances:
        input_string = input_string_from_chunks(utterance[DATA])
        parsing = engine.parse(input_string)
        utterance_metrics = compute_utterance_metrics(
            parsing, utterance, intent_name, slot_matching_lambda)
        if contains_errors(utterance_metrics):
            errors.append({
                "nlu_output": parsing,
                "expected_output": format_expected_output(intent_name,
                                                          utterance)
            })
        metrics = aggregate_metrics(metrics, utterance_metrics)
    return metrics, errors


def compute_utterance_metrics(parsing, utterance, utterance_intent,
                              slot_matching_lambda):
    if parsing["intent"] is not None:
        parsing_intent_name = parsing["intent"]["intentName"]
    else:
        # Use a string here to avoid having a None key in the metrics dict
        parsing_intent_name = NONE_INTENT_NAME

    parsed_slots = [] if parsing["slots"] is None else parsing["slots"]
    utterance_slots = filter(lambda chunk: SLOT_NAME in chunk, utterance[DATA])

    # initialize metrics
    intent_names = {parsing_intent_name, utterance_intent}
    slot_names = set(
        [(parsing_intent_name, s["slotName"]) for s in parsed_slots] +
        [(utterance_intent, u[SLOT_NAME]) for u in utterance_slots])

    metrics = {
        intent: {
            "intent": deepcopy(INITIAL_METRICS),
            "slots": dict(),
        } for intent in intent_names
    }

    for (intent_name, slot_name) in slot_names:
        metrics[intent_name]["slots"][slot_name] = deepcopy(INITIAL_METRICS)

    if parsing_intent_name == utterance_intent:
        metrics[parsing_intent_name]["intent"][TRUE_POSITIVE] += 1
    else:
        metrics[parsing_intent_name]["intent"][FALSE_POSITIVE] += 1
        metrics[utterance_intent]["intent"][FALSE_NEGATIVE] += 1
        return metrics

    # Check if expected slots have been parsed
    for slot in utterance_slots:
        slot_name = slot[SLOT_NAME]
        slot_metrics = metrics[utterance_intent]["slots"][slot_name]
        if any(s["slotName"] == slot_name and slot_matching_lambda(
                s["rawValue"], slot[TEXT]) for s in parsed_slots):
            slot_metrics[TRUE_POSITIVE] += 1
        else:
            slot_metrics[FALSE_NEGATIVE] += 1

    # Check if there are unexpected parsed slots
    for slot in parsed_slots:
        slot_name = slot["slotName"]
        slot_metrics = metrics[parsing_intent_name]["slots"][slot_name]
        if all(s[SLOT_NAME] != slot_name or not slot_matching_lambda(
                s[TEXT], slot["rawValue"]) for s in utterance_slots):
            slot_metrics[FALSE_POSITIVE] += 1
    return metrics


def aggregate_metrics(lhs_metrics, rhs_metrics):
    acc_metrics = deepcopy(lhs_metrics)
    for (intent, intent_metrics) in rhs_metrics.iteritems():
        if intent not in acc_metrics:
            acc_metrics[intent] = deepcopy(intent_metrics)
        else:
            acc_metrics[intent]["intent"] = add_count_metrics(
                acc_metrics[intent]["intent"], intent_metrics["intent"])
            acc_slot_metrics = acc_metrics[intent]["slots"]
            for (slot, slot_metrics) in intent_metrics["slots"].iteritems():
                if slot not in acc_slot_metrics:
                    acc_slot_metrics[slot] = deepcopy(slot_metrics)
                else:
                    acc_slot_metrics[slot] = add_count_metrics(
                        acc_slot_metrics[slot], slot_metrics)
    return acc_metrics


def add_count_metrics(lhs, rhs):
    return {
        TRUE_POSITIVE: lhs[TRUE_POSITIVE] + rhs[TRUE_POSITIVE],
        FALSE_POSITIVE: lhs[FALSE_POSITIVE] + rhs[FALSE_POSITIVE],
        FALSE_NEGATIVE: lhs[FALSE_NEGATIVE] + rhs[FALSE_NEGATIVE],
    }


def compute_precision_recall(metrics):
    for intent_metrics in metrics.values():
        prec_rec_metrics = _compute_precision_recall(intent_metrics["intent"])
        intent_metrics["intent"].update(prec_rec_metrics)
        for slot_metrics in intent_metrics["slots"].values():
            prec_rec_metrics = _compute_precision_recall(slot_metrics)
            slot_metrics.update(prec_rec_metrics)
    return metrics


def _compute_precision_recall(count_metrics):
    tp = count_metrics[TRUE_POSITIVE]
    fp = count_metrics[FALSE_POSITIVE]
    fn = count_metrics[FALSE_NEGATIVE]
    return {
        "precision": 0. if tp == 0 else float(tp) / float(tp + fp),
        "recall": 0. if tp == 0 else float(tp) / float(tp + fn),
    }


def contains_errors(utterance_metrics):
    for metrics in utterance_metrics.values():
        intent_metrics = metrics["intent"]
        if intent_metrics.get(FALSE_POSITIVE, 0) > 0:
            return True
        if intent_metrics.get(FALSE_NEGATIVE, 0) > 0:
            return True
        for slot_metrics in metrics["slots"].values():
            if slot_metrics.get(FALSE_POSITIVE, 0) > 0:
                return True
            if slot_metrics.get(FALSE_NEGATIVE, 0) > 0:
                return True
    return False


def format_expected_output(intent_name, utterance):
    char_index = 0
    ranges = []
    for chunk in utterance[DATA]:
        range_end = char_index + len(chunk[TEXT])
        ranges.append({"start": char_index, "end": range_end})
        char_index = range_end

    return {
        "input": "".join(chunk[TEXT] for chunk in utterance[DATA]),
        "intent": {
            "intentName": intent_name,
            "probability": 1.0
        },
        "slots": [
            {
                "rawValue": chunk[TEXT],
                "entity": chunk[ENTITY],
                "slotName": chunk[SLOT_NAME],
                "range": ranges[chunk_index]
            } for chunk_index, chunk in enumerate(utterance[DATA])
            if ENTITY in chunk
        ]
    }


def is_builtin_entity(entity_name):
    return entity_name.startswith("snips/")


def exact_match(lhs_slot, rhs_slot):
    return lhs_slot == rhs_slot
