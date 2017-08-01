from __future__ import unicode_literals

import requests

DEV = 'dev'
PROD = 'prod'
PROD_GATEWAY = 'https://external-gateway.snips.ai/v1/registry/intents'
DEV_GATEWAY = 'https://external-gateway-dev.snips.ai/v1/registry/intents'


def get_intents(grid, authors=None, languages=None, version="latest",
                intent_names=None):
    if authors is None:
        authors = ["intents@snips.ai"]
    if intent_names is None:
        intent_names = []
    if languages is None:
        languages = []

    if grid == DEV:
        url = DEV_GATEWAY
    elif grid == PROD:
        url = PROD_GATEWAY
    else:
        raise ValueError("Unknown grid: %s" % grid)

    if len(intent_names) == 1:
        url += "/" + intent_names[0]

    params = {
        "v": version,
        "customIntentData": True
    }
    if len(authors) == 1:
        params["email"] = authors[0]
    if len(languages) == 1:
        params["lang"] = languages[0]

    rep = requests.get(url, params)
    json_data = rep.json()
    intents = json_data.get("intents")
    filtered_intents = []
    for intent in intents:
        intent_name = intent["config"]["name"]
        intent_language = intent["config"]["language"]
        intent_author = intent["config"]["author"]
        engine_version = intent["config"]['engine']['version']
        if len(intent_names) > 0 and intent_name not in intent_names:
            continue
        if len(languages) > 0 and intent_language not in languages:
            continue
        if len(authors) > 0 and intent_author not in authors:
            continue
        intent['customIntentData']['snips_nlu_version'] = engine_version
        filtered_intents.append(intent)
    return filtered_intents


def create_intent_groups(config_intent_groups, intents):
    if config_intent_groups is None or len(config_intent_groups) == 0:
        return [{
            "name": intent["config"]["name"],
            "language": intent["config"]["language"],
            "intents": [intent]
        } for intent in intents]
    groups = []
    for config_group in config_intent_groups:
        group_name = config_group["name"]
        group_language = config_group["language"]
        group_intents = []
        for intent_name in config_group["intents"]:
            matching_intents = filter(
                lambda intent: intent["config"]["name"] == intent_name and
                               intent["config"]["language"] == group_language,
                intents)
            if len(matching_intents) == 0:
                raise AssertionError("Missing entry in the registry for "
                                     "intent '%s' in language '%s'"
                                     % (intent_name, group_language))
            group_intents.append(matching_intents[0])
        groups.append({
            "name": group_name,
            "language": group_language,
            "intents": group_intents,
        })
    return groups
