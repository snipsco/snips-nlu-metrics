from __future__ import unicode_literals

from copy import deepcopy

from snips_nlu.builtin_entities import is_builtin_entity
from snips_nlu.constants import (INTENTS, UTTERANCES, ENGINE_TYPE,
                                 CUSTOM_ENGINE, DATA, SLOT_NAME, TEXT)

from nlu_metrics.utils.dataset_utils import (input_string_from_chunks,
                                             get_stratified_utterances)

INITIAL_METRICS = {
    "true_positive": 0,
    "false_positive": 0,
    "false_negative": 0
}

NONE_INTENT_NAME = "null"


def create_k_fold_batches(dataset, k, max_training_utterances=None, seed=None):
    dataset = deepcopy(dataset)
    # Remove entity values in order to un-bias the cross validation
    for name, entity in dataset["entities"].iteritems():
        if is_builtin_entity(name):
            continue
        if entity["automatically_extensible"]:
            entity["data"] = []

    utterances = get_stratified_utterances(dataset, seed)
    if len(utterances) < k:
        raise AssertionError("The number of utterances ({0}) should be "
                             "greater than the fold size ({1}) in order to "
                             "compute metrics".format(len(utterances), k))
    if max_training_utterances is None:
        max_training_utterances = len(utterances)
    k_fold_batches = []
    batch_size = len(utterances) / k
    for batch_index in xrange(k):
        test_start = batch_index * batch_size
        test_end = (batch_index + 1) * batch_size
        train_utterances = utterances[0:test_start] + utterances[test_end:]
        train_utterances = train_utterances[0:max_training_utterances]
        test_utterances = utterances[test_start: test_end]
        train_dataset = deepcopy(dataset)
        train_dataset[INTENTS] = dict()
        for intent_name, utterance in train_utterances:
            if intent_name not in train_dataset[INTENTS]:
                train_dataset[INTENTS][intent_name] = {
                    ENGINE_TYPE: CUSTOM_ENGINE,
                    UTTERANCES: []
                }
            train_dataset[INTENTS][intent_name][UTTERANCES].append(
                deepcopy(utterance))
        k_fold_batches.append((train_dataset, test_utterances))
    return k_fold_batches


def compute_engine_metrics(engine, test_utterances, verbose=False):
    metrics = dict()
    for intent_name, utterance in test_utterances:
        input_string = input_string_from_chunks(utterance[DATA])
        parsing = engine.parse(input_string)
        utterance_metrics = compute_utterance_metrics(parsing, utterance,
                                                      intent_name, verbose)
        metrics = aggregate_metrics(metrics, utterance_metrics)
    return metrics


def compute_utterance_metrics(parsing, utterance, utterance_intent,
                              verbose=False):
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
        metrics[parsing_intent_name]["intent"]["true_positive"] += 1
    else:
        metrics[parsing_intent_name]["intent"]["false_positive"] += 1
        metrics[utterance_intent]["intent"]["false_negative"] += 1
        if verbose:
            intent_proba = parsing["intent"]["probability"] \
                if parsing["intent"] is not None else 0.0
            print("INTENT PROBA: %s" % intent_proba)
            print("Intent classification mismatch:\n"
                  "\tinput:         \t\"{0}\"\n"
                  "\tintent found:  \t{1} ({2:.0%})\n"
                  "\tcorrect intent:\t{3}\n"
                  .format(parsing["input"],
                          parsing_intent_name,
                          intent_proba,
                          utterance_intent))
        return metrics

    for slot in utterance_slots:
        slot_name = slot[SLOT_NAME]
        slot_metrics = metrics[utterance_intent]["slots"][slot_name]
        if any(s["slotName"] == slot_name and s["rawValue"] == slot[TEXT]
               for s in parsed_slots):
            slot_metrics["true_positive"] += 1
        else:
            slot_metrics["false_negative"] += 1
            if verbose:
                print("Slot filling mismatch (missing slot):\n"
                      "\tINPUT:     \t\"{0}\"\n"
                      "\tSLOT:      \t{1}\n"
                      "\tSLOT VALUE:\t\"{2}\"\n"
                      .format(parsing["input"], slot_name, slot[TEXT]))

    for slot in parsed_slots:
        slot_name = slot["slotName"]
        slot_metrics = metrics[parsing_intent_name]["slots"][slot_name]
        if all(s[SLOT_NAME] != slot_name or s[TEXT] != slot["rawValue"]
               for s in utterance_slots):
            slot_metrics["false_positive"] += 1
            if verbose:
                print("Slot filling mismatch (unexpected slot):\n"
                      "\tINPUT:     \t\"{0}\"\n"
                      "\tSLOT:      \t{1}\n"
                      "\tSLOT VALUE:\t\"{2}\"\n"
                      .format(parsing["input"], slot_name, slot["rawValue"]))

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
        "true_positive": lhs["true_positive"] + rhs["true_positive"],
        "false_positive": lhs["false_positive"] + rhs["false_positive"],
        "false_negative": lhs["false_negative"] + rhs["false_negative"],
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
    tp = count_metrics["true_positive"]
    fp = count_metrics["false_positive"]
    fn = count_metrics["false_negative"]
    return {
        "precision": 0. if tp == 0 else float(tp) / float(tp + fp),
        "recall": 0. if tp == 0 else float(tp) / float(tp + fn),
    }
