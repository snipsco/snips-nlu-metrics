from __future__ import unicode_literals

import io
import json
import os
from copy import deepcopy

from datetime import datetime


def save_metrics_into_json(metrics, grid, language, intent_group,
                           max_utterances, k_fold_size, metrics_dir,
                           timestamp):
    metrics_path = os.path.join(
        metrics_dir, grid, language, intent_group,
        "k_fold_size_%s" % k_fold_size, "max_utterances_%s" % max_utterances,
        "%d.json" % timestamp)

    directory = os.path.dirname(metrics_path)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with io.open(metrics_path, mode='w', encoding="utf8") as f:
        f.write(json.dumps(metrics, indent=4).decode(encoding="utf8"))


def save_metrics_into_db(db, metrics, grid, language, intent_group,
                         authors, max_utterances, k_fold_size, timestamp):
    slot_metrics_collection = db['slot-metrics']
    intent_metrics_collection = db['intent-metrics']

    base_entry = {
        "grid": grid,
        "language": language,
        "intent_group": intent_group,
        "max_utterances": max_utterances,
        "k_fold_size": k_fold_size,
        "datetime": datetime.utcfromtimestamp(timestamp),
    }

    for intent_name, intent_metrics in metrics.iteritems():
        intent_metrics_entry = deepcopy(base_entry)
        intent_metrics_entry.update({
            "intent_name": intent_name,
            "author": authors.get(intent_name, None),
            "intent_metrics": metrics[intent_name]["intent"],
            "intent_utterances": metrics[intent_name]["intent_utterances"]
        })
        intent_metrics_collection.insert_one(intent_metrics_entry)
        for slot_name, slot_metrics in \
                metrics[intent_name]["slots"].iteritems():
            slot_metrics_entry = deepcopy(base_entry)
            slot_metrics_entry.update({
                "intent_name": intent_name,
                "author": authors.get(intent_name, None),
                "slot_name": slot_name,
                "slot_metrics": slot_metrics
            })
            slot_metrics_collection.insert_one(slot_metrics_entry)
