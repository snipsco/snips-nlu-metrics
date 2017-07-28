from __future__ import unicode_literals

from snips_nlu.constants import TEXT, INTENTS, UTTERANCES
import random


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
