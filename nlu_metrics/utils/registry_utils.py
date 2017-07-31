from __future__ import unicode_literals

import glob
import io
import json
import os
from os import mkdir

import requests

DEV = 'dev'
PROD = 'prod'
PROD_GATEWAY = 'https://external-gateway.snips.ai/v1/registry/intents'
DEV_GATEWAY = 'https://external-gateway-dev.snips.ai/v1/registry/intents'


def get_intents(grid, email, language=None, version='latest', intent_name=None,
                persisting_dir_path=None, force_fetch=None):
    if force_fetch is None:
        force_fetch = False
    if grid == DEV:
        url = DEV_GATEWAY
    elif grid == PROD:
        url = PROD_GATEWAY
    else:
        raise ValueError("Unknown grid: %s" % grid)

    intents = []
    if persisting_dir_path is not None:
        if not os.path.exists(persisting_dir_path):
            mkdir(persisting_dir_path)
        if not force_fetch:
            intents_files = glob.glob1(persisting_dir_path, '*.json')
            for path in intents_files:
                with io.open(path, encoding='utf8') as f:
                    intent = json.load(f)
                if language is not None \
                        and intent['config']['language'] != language:
                    continue
                if intent_name is not None \
                        and intent['config']['name'] != intent_name:
                    continue
                if email != intent['config']['author']:
                    continue
                intents.append(intent)

    if len(intents) > 0:
        return intents

    if intent_name is not None:
        url += '/' + intent_name

    params = {
        'v': version,
        'email': email,
        'customIntentData': True
    }
    if language is not None:
        params['lang'] = language

    rep = requests.get(url, params)
    json_data = rep.json()
    intents = json_data.get('intents')
    for intent in intents:
        engine_version = intent['config']['engine']['version']
        intent['customIntentData']['snips_nlu_version'] = engine_version

    if persisting_dir_path is not None:
        for intent in intents:
            intent_path = os.path.join(persisting_dir_path,
                                       '%s.json' % intent['config']['name'])
            with io.open(intent_path, mode='w', encoding='utf8') as f:
                f.write(json.dumps(intent, indent=4).decode(encoding='utf8'))
    return intents
