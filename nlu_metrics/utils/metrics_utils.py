from __future__ import unicode_literals

from copy import deepcopy

from snips_nlu.constants import INTENTS, UTTERANCES, ENGINE_TYPE, \
    CUSTOM_ENGINE, DATA, SLOT_NAME, PARSED_INTENT, INTENT_NAME, PARSED_SLOTS, \
    MATCH_RANGE, TEXT

from nlu_metrics.utils.dataset_utils import input_string_from_chunks


def create_k_fold_batches(dataset, k):
    utterances = [
        (intent_name, utterance, i)
        for intent_name, intent_data in dataset[INTENTS].iteritems()
        for i, utterance in enumerate(intent_data[UTTERANCES])
    ]
    utterances = sorted(utterances, key=lambda u: u[2])
    utterances = [(intent_name, utterance) for (intent_name, utterance, _) in
                  utterances]
    nb_utterances = len(utterances)
    k_fold_batches = []
    batch_size = nb_utterances / k
    for batch_index in xrange(k):
        test_start = batch_index * batch_size
        test_end = (batch_index + 1) * batch_size
        train_utterances = utterances[0:test_start] + utterances[test_end:]
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


def compute_engine_metrics(engine, test_utterances):
    metrics = {
        "intents": dict(),
        "slots": dict()
    }
    for intent_name, utterance in test_utterances:
        input_string = input_string_from_chunks(utterance[DATA])
        parsing = engine.parse(input_string)
        utterance_metrics = compute_utterance_metrics(parsing, utterance,
                                                      intent_name)
        metrics = update_metrics(metrics, utterance_metrics)


def compute_utterance_metrics(parsing, utterance, utterance_intent):
    metrics = {
        "intents": dict(),
        "slots": dict()
    }

    if parsing["intent"] is not None:
        parsing_intent_name = parsing["intent"]["intentName"]
    else:
        parsing_intent_name = None

    parsed_slots = [] if parsing["slots"] is None else parsing["slots"]
    utterance_slots = filter(lambda chunk: SLOT_NAME in chunk, utterance[DATA])

    # initialize metrics
    intent_names = {parsing_intent_name, utterance_intent}
    slot_names = set([s["slotName"] for s in parsed_slots] +
                     [u[SLOT_NAME] for u in utterance_slots])

    # precision = true_positive / (true_positive + false_positive)
    # recall = true_positive / (true_positive + false_negative)

    initial_metrics = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0
    }

    for intent in intent_names:
        metrics["intents"][intent] = deepcopy(initial_metrics)

    for slot in slot_names:
        metrics["slots"][slot] = deepcopy(initial_metrics)

    if parsing_intent_name == utterance_intent:
        metrics["intents"][parsing_intent_name]["true_positive"] += 1
    else:
        metrics["intents"][parsing_intent_name]["false_positive"] += 1
        metrics["intents"][utterance_intent]["false_negative"] += 1
        return metrics

    for slot in utterance_slots:
        slot_name = slot[SLOT_NAME]
        if any(s["slotName"] == slot_name and s["rawValue"] == slot[TEXT]
               for s in parsed_slots):
            metrics["slots"][slot_name]["true_positive"] += 1
        else:
            metrics["slots"][slot_name]["false_negative"] += 1

    for slot in parsed_slots:
        slot_name = slot["slotName"]
        if all(s[SLOT_NAME] != slot_name or s[TEXT] != slot["rawValue"]
               for s in utterance_slots):
            metrics["slots"][slot_name]["false_positive"] += 1

    return metrics


def update_metrics(dataset_metrics, utterance_metrics):
    for (intent, intent_metrics) in utterance_metrics["intents"].iteritems():
        if intent not in dataset_metrics["intents"]:
            dataset_metrics["intents"][intent] = deepcopy(intent_metrics)
        else:
            dataset_metrics["intents"][intent] = add_metrics(
                dataset_metrics["intents"][intent], intent_metrics)

    for (slot_name, slot_metrics) in utterance_metrics["slots"].iteritems():
        if slot_name not in dataset_metrics["slots"]:
            dataset_metrics["slots"][slot_name] = deepcopy(slot_metrics)
        else:
            dataset_metrics["slots"][slot_name] = add_metrics(
                dataset_metrics["slots"][slot_name], slot_metrics)
    return dataset_metrics


def add_metrics(lhs, rhs):
    return {
        "true_positive": lhs["true_positive"] + rhs["true_positive"],
        "false_positive": lhs["false_positive"] + rhs["false_positive"],
        "false_negative": lhs["false_negative"] + rhs["false_negative"],
    }
