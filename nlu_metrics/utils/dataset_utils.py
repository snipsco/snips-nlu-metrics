from __future__ import unicode_literals
from __future__ import absolute_import

from copy import deepcopy

from .constants import TEXT


def input_string_from_chunks(chunks):
    return "".join(chunk[TEXT] for chunk in chunks)


def get_utterances_subset(utterances, ratio):
    utterances_dict = dict()
    for (intent_name, utterance) in utterances:
        if intent_name not in utterances_dict:
            utterances_dict[intent_name] = []
        utterances_dict[intent_name].append(deepcopy(utterance))

    utterances_subset = []
    for (intent_name, utterances) in utterances_dict.items():
        nb_utterances = int(ratio * len(utterances))
        utterances_subset += [(intent_name, u)
                              for u in utterances[:nb_utterances]]
    return utterances_subset
