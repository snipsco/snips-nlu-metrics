from __future__ import unicode_literals

from copy import deepcopy


def save_metrics_into_db(db, metrics, grid, language, intent_group,
                         authors, max_utterances, k_fold_size, datetime):
    slot_metrics_collection = db['slot-metrics']
    intent_metrics_collection = db['intent-metrics']

    base_entry = {
        "grid": grid,
        "language": language,
        "intent_group": intent_group,
        "max_utterances": max_utterances,
        "k_fold_size": k_fold_size,
        "datetime": datetime,
    }

    for intent_name, intent_metrics in metrics.iteritems():
        intent_metrics_entry = deepcopy(base_entry)
        intent_metrics_entry.update({
            "intent_name": intent_name,
            "author": authors[intent_name],
            "intent_metrics": metrics[intent_name]["intent"],
            "intent_utterances": metrics[intent_name]["intent_utterances"]
        })
        intent_metrics_collection.insert_one(intent_metrics_entry)
        for slot_name, slot_metrics in \
                metrics[intent_name]["slots"].iteritems():
            slot_metrics_entry = deepcopy(base_entry)
            slot_metrics_entry.update({
                "intent_name": intent_name,
                "author": authors[intent_name],
                "slot_name": slot_name,
                "slot_metrics": slot_metrics
            })
            slot_metrics_collection.insert_one(slot_metrics_entry)
