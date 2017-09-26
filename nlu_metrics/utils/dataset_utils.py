from __future__ import unicode_literals

import random
from copy import deepcopy

from constants import TEXT, UTTERANCES, INTENTS


def input_string_from_chunks(chunks):
    return "".join(chunk[TEXT] for chunk in chunks)


def get_utterances_subset(utterances, ratio):
    utterances_dict = dict()
    for (intent_name, utterance) in utterances:
        if intent_name not in utterances_dict:
            utterances_dict[intent_name] = []
        utterances_dict[intent_name].append(deepcopy(utterance))

    utterances_subset = []
    for (intent_name, utterances) in utterances_dict.iteritems():
        nb_utterances = int(ratio * len(utterances))
        utterances_subset += [(intent_name, u)
                              for u in utterances[:nb_utterances]]
    return utterances_subset


def get_stratified_utterances(dataset, seed, shuffle=True):
    if shuffle:
        shuffle_dataset(dataset, seed)
    utterances = [
        (intent_name, utterance, i)
        for intent_name, intent_data in dataset[INTENTS].iteritems()
        for i, utterance in enumerate(intent_data[UTTERANCES])
    ]
    utterances = sorted(utterances, key=lambda u: u[2])
    utterances = [(intent_name, utterance) for (intent_name, utterance, _) in
                  utterances]
    return utterances


def shuffle_dataset(dataset, seed):
    random.seed(seed)
    for intent_data in dataset[INTENTS].values():
        random.shuffle(intent_data[UTTERANCES])


def create_nlu_dataset(registry_intents):
    language = registry_intents[0]["config"]["language"]
    nlu_dataset = {
        "language": language,
        "snips_nlu_version": "0.1.0",
        INTENTS: {
            intent["config"]["displayName"]: {
                "engineType": "regex",
                "utterances": intent["customIntentData"]["utterances"]
            } for intent in registry_intents
        },
        "entities": {
            entity_name: entity
            for intent in registry_intents for entity_name, entity in
            intent["customIntentData"]["entities"].iteritems()
        }
    }
    return nlu_dataset
