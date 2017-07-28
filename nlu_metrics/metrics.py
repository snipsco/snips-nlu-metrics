from __future__ import unicode_literals

import io
import json
import time
import os

from nlu_metrics.utils.dataset_utils import (get_stratified_utterances,
                                             format_registry_dataset)
from nlu_metrics.utils.dependency_utils import update_nlu_packages
from nlu_metrics.utils.metrics_utils import (create_k_fold_batches,
                                             compute_engine_metrics,
                                             aggregate_metrics,
                                             compute_precision_recall)
from nlu_metrics.utils.nlu_engine_utils import get_trained_nlu_engine
from nlu_metrics.utils.registry_utils import get_intents


def compute_cross_val_metrics(language, dataset, snips_nlu_version,
                              snips_nlu_rust_version, k_fold_size=5,
                              max_utterances=None):
    """Compute the main NLU metrics on the dataset using cross validation

    :param language: str
    :param dataset: dict
    :param snips_nlu_version: optional str, semver, None --> use local version
    :param snips_nlu_rust_version: str, semver, None --> use local version
    :param k_fold_size: int, number of folds to use for cross validation
    :param max_utterances: int, max number of utterances to use for training
    :return: dict containing the metrics

    """
    update_nlu_packages(snips_nlu_version=snips_nlu_version,
                        snips_nlu_rust_version=snips_nlu_rust_version)

    batches = create_k_fold_batches(dataset, k=k_fold_size,
                                    max_training_utterances=max_utterances)

    global_metrics = dict()

    for batch_index, (train_dataset, test_utterances) in enumerate(batches):
        engine = get_trained_nlu_engine(language, train_dataset)
        batch_metrics = compute_engine_metrics(engine, test_utterances)
        global_metrics = aggregate_metrics(global_metrics, batch_metrics)

    global_metrics = compute_precision_recall(global_metrics)

    return global_metrics


def compute_train_test_metrics(language, train_dataset, test_dataset,
                               snips_nlu_version, snips_nlu_rust_version):
    """Compute the main NLU metrics on `test_dataset` after having trained on
    `trained_dataset`

    :param language: str
    :param train_dataset: dict, dataset used for training
    :param test_dataset: dict, dataset used for testing
    :param snips_nlu_version: str, semver, None --> use local version
    :param snips_nlu_rust_version: str, semver, None --> use local version
    :return: dict containing the metrics
    """
    update_nlu_packages(snips_nlu_version=snips_nlu_version,
                        snips_nlu_rust_version=snips_nlu_rust_version)
    engine = get_trained_nlu_engine(language, train_dataset)
    utterances = get_stratified_utterances(test_dataset, seed=None,
                                           shuffle=False)
    metrics = compute_engine_metrics(engine, utterances)
    metrics = compute_precision_recall(metrics)
    return metrics


def compute_batch_metrics(metrics_config):
    timestamp = time.time()
    grid = metrics_config["grid"]
    snips_nlu_version = metrics_config["snips_nlu_version"]
    snips_nlu_rust_version = metrics_config["snips_nlu_rust_version"]

    print("Fetching intents on %s" % grid)
    intents = get_intents(
        grid=grid,
        email=metrics_config["email_author"],
        language=metrics_config.get("language", None),
        version=metrics_config["version"],
        intent_name=metrics_config.get("intent_name", None),
        persisting_dir_path=metrics_config.get("intents_data_dir", None),
        force_fetch=metrics_config["force_fetch"])
    print("%s intents fetched" % len(intents))

    for intent in intents:
        language = intent["config"]["language"]
        intent_name = intent["config"]["name"]

        print("Computing metrics for intent '%s' in language '%s'"
              % (intent_name, language))

        nb_utterances = len(intent["customIntentData"]["utterances"])
        if nb_utterances < min(metrics_config["k_fold_sizes"]):
            print("Skipping intent because number of utterances is too "
                  "low (%s)" % nb_utterances)
            continue
        dataset = format_registry_dataset(intent["customIntentData"],
                                          intent_name, language)

        for max_utterances in metrics_config["max_utterances"]:
            print("\tmax utterances: %d" % max_utterances)
            for k_fold_size in metrics_config["k_fold_sizes"]:
                print("\t\tk_fold_size: %d" % k_fold_size)
                metrics = compute_cross_val_metrics(
                    language, dataset,
                    snips_nlu_version=snips_nlu_version,
                    snips_nlu_rust_version=snips_nlu_rust_version,
                    k_fold_size=k_fold_size,
                    max_utterances=max_utterances)
                save_metrics(metrics, language, intent_name, intent_name,
                             max_utterances, k_fold_size,
                             metrics_config["metrics_dir"], timestamp)


def save_metrics(metrics, language, intent_group, intent_name, max_utterances,
                 k_fold_size, metrics_dir, timestamp):
    metrics_path = os.path.join(
        metrics_dir, language, intent_group, intent_name,
        "max_utterances_%s" % max_utterances,
        "k_fold_size_%s" % k_fold_size, "%d.json" % timestamp)

    directory = os.path.dirname(metrics_path)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with io.open(metrics_path, mode='w', encoding="utf8") as f:
        f.write(json.dumps(metrics, indent=4).decode(encoding="utf8"))
