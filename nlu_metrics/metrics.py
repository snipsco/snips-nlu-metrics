from __future__ import unicode_literals

import argparse
import datetime
import io
import json

from nlu_metrics.database import save_metrics_into_db
from nlu_metrics.utils.dataset_utils import (get_stratified_utterances,
                                             create_nlu_dataset)
from nlu_metrics.utils.dependency_utils import update_nlu_packages
from nlu_metrics.utils.metrics_utils import (create_k_fold_batches,
                                             compute_engine_metrics,
                                             aggregate_metrics,
                                             compute_precision_recall)
from nlu_metrics.utils.nlu_engine_utils import get_trained_nlu_engine
from nlu_metrics.utils.registry_utils import get_intents, create_intent_groups


def compute_cross_val_metrics(dataset, snips_nlu_version,
                              snips_nlu_rust_version, k_fold_size=5,
                              max_utterances=None):
    """Compute the main NLU metrics on the dataset using cross validation

    :param dataset: dict
    :param snips_nlu_version: optional str, semver, None --> use local version
    :param snips_nlu_rust_version: str, semver, None --> use local version
    :param k_fold_size: int, number of folds to use for cross validation
    :param max_utterances: int, max number of utterances to use for training
    :return: dict containing the metrics

    """
    nb_utterances = sum(len(intent["utterances"])
                        for intent in dataset["intents"].values())
    if nb_utterances < k_fold_size:
        print("Skipping group because number of utterances is too "
              "low (%s)" % nb_utterances)
        return None
    update_nlu_packages(snips_nlu_version, snips_nlu_rust_version)
    batches = create_k_fold_batches(dataset, k_fold_size, max_utterances)
    global_metrics = dict()

    for batch_index, (train_dataset, test_utterances) in enumerate(batches):
        try:
            engine = get_trained_nlu_engine(train_dataset)
        except Exception as e:
            print("Skipping group because of training error: %s" % e.message)
            return None
        batch_metrics = compute_engine_metrics(engine, test_utterances)
        global_metrics = aggregate_metrics(global_metrics, batch_metrics)

    global_metrics = compute_precision_recall(global_metrics)

    for intent, metrics in global_metrics.iteritems():
        if intent is None:
            metrics["intent_utterances"] = 0
        else:
            metrics["intent_utterances"] = len(
                dataset["intents"][intent]["utterances"])

    return global_metrics


def compute_train_test_metrics(train_dataset, test_dataset,
                               snips_nlu_version, snips_nlu_rust_version):
    """Compute the main NLU metrics on `test_dataset` after having trained on
    `trained_dataset`

    :param train_dataset: dict, dataset used for training
    :param test_dataset: dict, dataset used for testing
    :param snips_nlu_version: str, semver, None --> use local version
    :param snips_nlu_rust_version: str, semver, None --> use local version
    :return: dict containing the metrics
    """
    update_nlu_packages(snips_nlu_version=snips_nlu_version,
                        snips_nlu_rust_version=snips_nlu_rust_version)
    engine = get_trained_nlu_engine(train_dataset)
    utterances = get_stratified_utterances(test_dataset, seed=None,
                                           shuffle=False)
    metrics = compute_engine_metrics(engine, utterances)
    metrics = compute_precision_recall(metrics)
    return metrics


def run_and_save_registry_metrics(metrics_config):
    if isinstance(metrics_config, (str, unicode)):
        with io.open(metrics_config, encoding="utf8") as f:
            config = json.load(f)
    else:
        config = metrics_config
    timestamp = datetime.datetime.utcnow()
    grid = config["grid"]
    snips_nlu_version = config["snips_nlu_version"]
    snips_nlu_rust_version = config["snips_nlu_rust_version"]
    authors = config["authors"]
    languages = config.get("languages", None)
    version = config["version"]
    intent_names = config.get("intent_names", None)
    intent_groups = config.get("intent_groups", None)
    print("Fetching intents on %s" % grid)
    intents = get_intents(grid, authors, languages, version, intent_names)
    print("%s intents fetched" % len(intents))
    intent_groups = create_intent_groups(intent_groups, intents)

    for group in intent_groups:
        language = group["language"]
        group_name = group["name"]
        dataset = create_nlu_dataset(group["intents"])
        authors = {intent["config"]["name"]: intent["config"]["author"]
                   for intent in group["intents"]}
        print("Computing metrics for intent group '%s' in language '%s'"
              % (group_name, language))

        for k_fold_size in config["k_fold_sizes"]:
            print("\tk_fold_size: %d" % k_fold_size)
            for max_utterances in config["max_utterances"]:
                print("\t\tmax utterances: %d" % max_utterances)
                metrics = compute_cross_val_metrics(
                    dataset, snips_nlu_version, snips_nlu_rust_version,
                    k_fold_size, max_utterances)
                if metrics is None:
                    break
                save_metrics_into_db(
                    metrics, grid, language, group_name, authors,
                    max_utterances, k_fold_size, timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the metrics config file")
    args = parser.parse_args()
    run_and_save_registry_metrics(args.config_path)
