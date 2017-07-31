from __future__ import unicode_literals

import random

from snips_nlu.constants import TEXT, INTENTS, UTTERANCES


def input_string_from_chunks(chunks):
    return "".join(chunk[TEXT] for chunk in chunks)


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
        "intents": {
            intent["config"]["name"]: {
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
