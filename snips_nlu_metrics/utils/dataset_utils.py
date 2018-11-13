from __future__ import absolute_import, unicode_literals

from collections import defaultdict
from copy import deepcopy

from future.utils import iteritems, itervalues

from snips_nlu_metrics.utils.constants import (
    DATA, ENTITIES, ENTITY, INTENTS, SYNONYMS, TEXT, USE_SYNONYMS, UTTERANCES,
    VALUE)


def input_string_from_chunks(chunks):
    return "".join(chunk[TEXT] for chunk in chunks)


def get_utterances_subset(utterances, ratio):
    utterances_dict = dict()
    for (intent_name, utterance) in utterances:
        if intent_name not in utterances_dict:
            utterances_dict[intent_name] = []
        utterances_dict[intent_name].append(deepcopy(utterance))

    utterances_subset = []
    for (intent_name, utterances) in iteritems(utterances_dict):
        nb_utterances = int(ratio * len(utterances))
        utterances_subset += [(intent_name, u)
                              for u in utterances[:nb_utterances]]
    return utterances_subset


def is_builtin_entity(entity_name):
    return entity_name.startswith("snips/")


def get_declared_entities_values(dataset):
    existing_entities = dict()
    for entity_name, entity in iteritems(dataset[ENTITIES]):
        if is_builtin_entity(entity_name):
            continue
        ents = set()
        for data in entity[DATA]:
            ents.add(data[VALUE])
            if entity[USE_SYNONYMS]:
                for s in data[SYNONYMS]:
                    ents.add(s)
        existing_entities[entity_name] = ents
    return existing_entities


def get_intent_utterances_entities_value(dataset):
    existing_entities = defaultdict(set)
    for intent in itervalues(dataset[INTENTS]):
        for u in intent[UTTERANCES]:
            for chunk in u[DATA]:
                if ENTITY not in chunk or is_builtin_entity(chunk[ENTITY]):
                    continue
                existing_entities[chunk[ENTITY]].add(chunk[TEXT])
    return existing_entities


def make_entity(value, synonyms):
    return {"value": value, "synonyms": synonyms}


def update_entities_with_utterances(dataset):
    dataset = deepcopy(dataset)

    declared_entities = get_declared_entities_values(dataset)
    intent_entities = get_intent_utterances_entities_value(dataset)

    for entity_name, existing_entities in iteritems(declared_entities):
        for entity_value in intent_entities.get(entity_name, []):
            if entity_value not in existing_entities:
                dataset[ENTITIES][entity_name][DATA].append(
                    make_entity(entity_value, []))

    return dataset
