from __future__ import unicode_literals

import requests

DEV = 'dev'
PROD = 'prod'
PROD_GATEWAY = 'https://external-gateway.snips.ai/v1/registry/intents'
DEV_GATEWAY = 'https://external-gateway-dev.snips.ai/v1/registry/intents'

SNIPS_EMAIL = "intents@snips.ai"


def get_intents(grid, authors=None, languages=None, intent_groups=None,
                api_token=None):
    intent_names = set()
    if authors is None or len(authors) == 0:
        authors = [SNIPS_EMAIL]
    if intent_groups is not None:
        intent_names = set(intent for group in intent_groups
                           for intent in group["intents"])
    if languages is None:
        languages = []

    if grid == DEV:
        url = DEV_GATEWAY
    elif grid == PROD:
        url = PROD_GATEWAY
    else:
        raise ValueError("Unknown grid: %s" % grid)

    if len(intent_names) == 1:
        url += "/" + intent_names.__iter__().next()

    params = {
        "v": "latest",
        "customIntentData": True
    }
    if len(languages) == 1:
        params["lang"] = languages[0]

    filtered_intents = []
    for author in authors:
        params["email"] = author
        if author != SNIPS_EMAIL and api_token is not None:
            params["apiToken"] = api_token
        rep = requests.get(url, params)
        json_data = rep.json()
        intents = json_data.get("intents")
        for intent in intents:
            intent_name = intent["config"]["displayName"]
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
            "name": intent["config"]["displayName"],
            "language": intent["config"]["language"],
            "intents": [intent]
        } for intent in intents]
    groups = []
    languages = set(intent["config"]["language"] for intent in intents)
    for config_group in config_intent_groups:
        group_name = config_group["name"]
        for language in languages:
            group_intents = [
                intent for intent in intents if
                intent["config"]["language"] == language and
                intent["config"]["displayName"] in config_group["intents"]]
            if len(group_intents) > 0:
                groups.append({
                    "name": group_name,
                    "language": language,
                    "intents": group_intents,
                })
    return groups
