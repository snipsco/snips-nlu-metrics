from __future__ import unicode_literals

import argparse
import io
import json
import time

from pymongo import MongoClient

from nlu_metrics.metrics_io import save_metrics_into_db
from nlu_metrics.metrics_io import save_metrics_into_json
from nlu_metrics.utils.dataset_utils import (get_stratified_utterances,
                                             create_nlu_dataset)
from nlu_metrics.utils.dependency_utils import (DEFAULT_TRAINING_VERSION,
                                                DEFAULT_INFERENCE_VERSION)
from nlu_metrics.utils.dependency_utils import update_nlu_packages
from nlu_metrics.utils.metrics_utils import (create_k_fold_batches,
                                             compute_engine_metrics,
                                             aggregate_metrics,
                                             compute_precision_recall)
from nlu_metrics.utils.nlu_engine_utils import get_trained_nlu_engine
from nlu_metrics.utils.registry_utils import get_intents, create_intent_groups


def compute_cross_val_metrics(
        dataset,
        snips_nlu_version=DEFAULT_TRAINING_VERSION,
        snips_nlu_rust_version=DEFAULT_INFERENCE_VERSION,
        training_engine_class=None,
        k_fold_size=5,
        max_utterances=None):
    """Compute the main NLU metrics on the dataset using cross validation

    :param dataset: dict or str, dataset or path to dataset
    :param snips_nlu_version: str, semver
    :param snips_nlu_rust_version: str, semver
    :param training_engine_class: SnipsNLUEngine class, if `None` then the
        engine used for training is created with the specified
        `snips_nlu_version`
    :param k_fold_size: int, number of folds to use for cross validation
    :param max_utterances: int, max number of utterances to use for training
    :return: dict containing the metrics

    """

    if isinstance(dataset, (str, unicode)):
        with io.open(dataset, encoding="utf8") as f:
            dataset = json.load(f)

    nb_utterances = {intent: len(data["utterances"])
                     for intent, data in dataset["intents"].iteritems()}
    total_utterances = sum(nb_utterances.values())
    should_skip = total_utterances < k_fold_size or (
        max_utterances is not None and total_utterances < max_utterances)
    if should_skip:
        print("Skipping group because number of utterances is too "
              "low (%s)" % total_utterances)
        return None
    update_nlu_packages(snips_nlu_version, snips_nlu_rust_version)
    batches = create_k_fold_batches(dataset, k_fold_size, max_utterances)
    global_metrics = dict()

    for batch_index, (train_dataset, test_utterances) in enumerate(batches):
        try:
            engine = get_trained_nlu_engine(train_dataset,
                                            training_engine_class)
        except Exception as e:
            print("Skipping group because of training error: %s" % e.message)
            return None
        batch_metrics = compute_engine_metrics(engine, test_utterances)
        global_metrics = aggregate_metrics(global_metrics, batch_metrics)

    global_metrics = compute_precision_recall(global_metrics)

    for intent, metrics in global_metrics.iteritems():
        metrics["intent_utterances"] = nb_utterances.get(intent, 0)

    return global_metrics


def compute_train_test_metrics(
        train_dataset,
        test_dataset,
        snips_nlu_version=DEFAULT_TRAINING_VERSION,
        snips_nlu_rust_version=DEFAULT_INFERENCE_VERSION,
        training_engine_class=None,
        verbose=False):
    """Compute the main NLU metrics on `test_dataset` after having trained on
    `trained_dataset`

    :param train_dataset: dict or str, dataset or path to dataset used for
        training
    :param test_dataset: dict or str, dataset or path to dataset used for
        testing
    :param snips_nlu_version: str, semver
    :param snips_nlu_rust_version: str, semver
    :param training_engine_class: SnipsNLUEngine class, if `None` then the
        engine used for training is created with the specified
        `snips_nlu_version`
    :param verbose: if `True` it will print prediction errors
    :return: dict containing the metrics
    """
    if isinstance(train_dataset, (str, unicode)):
        with io.open(train_dataset, encoding="utf8") as f:
            train_dataset = json.load(f)

    if isinstance(test_dataset, (str, unicode)):
        with io.open(test_dataset, encoding="utf8") as f:
            test_dataset = json.load(f)

    update_nlu_packages(snips_nlu_version=snips_nlu_version,
                        snips_nlu_rust_version=snips_nlu_rust_version)
    engine = get_trained_nlu_engine(train_dataset, training_engine_class)
    utterances = get_stratified_utterances(test_dataset, seed=None,
                                           shuffle=False)
    metrics = compute_engine_metrics(engine, utterances, verbose)
    metrics = compute_precision_recall(metrics)
    return metrics


def run_and_save_registry_metrics(
        grid,
        authors,
        k_fold_sizes,
        max_utterances,
        snips_nlu_version=DEFAULT_TRAINING_VERSION,
        snips_nlu_rust_version=DEFAULT_INFERENCE_VERSION,
        training_engine_class=None,
        api_token=None,
        languages=None,
        intent_groups=None,
        output_dir=None,
        mongo_host="localhost",
        mongo_port=27017):
    """Compute metrics on registry intents using cross validation

    :param grid: str, "dev" or "prod" --> backend environment to use
    :param snips_nlu_version: str, semver, version to use for training
    :param snips_nlu_rust_version: str, semver, version to use for inference
    :param training_engine_class: SnipsNLUEngine class, if `None` then the
        engine used for training is created with the specified
        `snips_nlu_version`
    :param authors: list, intent author emails that will be used to filter out
        intents
    :param max_utterances: list, training sizes to use for cross validation
    :param k_fold_sizes: list, fold sizes to use for cross validation
    :param api_token: str, must be specified when computing metrics on
        private intents
    :param languages: list, intent languages that will be used to filter out
        intents
    :param intent_groups: list, groups combining multiple intents.
        It allows to compute metrics on bundles or assistants, for instance:

        >>> groups = [
        ...   {
        ...     "name": "Creative Work",
        ...     "intents": [
        ...       "SearchCreativeWork",
        ...       "SearchCreativeWorkSection",
        ...       "SuspendCreativeWork",
        ...       "StartCreativeWork",
        ...       "StopCreativeWork",
        ...       "ResumeCreativeWork"
        ...     ]
        ...   }
        ... ]

    :param output_dir: str, optional, if not `None` then the metrics will be
        saved in json files under the `output_dir` directory
    :param mongo_host: str, optional, used to persist in a mongo db
    :param mongo_port: str, optional, used to persist in a mongo db
    """
    if mongo_port is not None and mongo_host is not None:
        mongo_client = MongoClient(mongo_host, mongo_port)
        db = mongo_client['nlu-metrics']
    else:
        db = None
    current_time = int(time.time())
    print("Fetching intents on %s" % grid)
    intents = get_intents(grid, authors, languages, intent_groups, api_token)
    print("%s intents fetched" % len(intents))
    intent_groups = create_intent_groups(intent_groups, intents)
    for group in intent_groups:
        language = group["language"]
        group_name = group["name"]
        dataset = create_nlu_dataset(group["intents"])
        authors = {intent["config"]["displayName"]: intent["config"]["author"]
                   for intent in group["intents"]}
        print("Computing metrics for intent group '%s' in language '%s'"
              % (group_name, language))

        for k_fold_size in k_fold_sizes:
            print("\tk_fold_size: %d" % k_fold_size)
            for train_utterances in max_utterances:
                print("\t\tmax utterances: %d" % train_utterances)
                metrics = compute_cross_val_metrics(
                    dataset, snips_nlu_version, snips_nlu_rust_version,
                    training_engine_class, k_fold_size, train_utterances)
                if metrics is None:
                    break
                if db is not None:
                    save_metrics_into_db(
                        db, metrics, grid, language, group_name, authors,
                        train_utterances, k_fold_size, current_time)
                if output_dir is not None:
                    save_metrics_into_json(
                        metrics, grid, language, group_name, train_utterances,
                        k_fold_size, output_dir, current_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the metrics config file")
    args = parser.parse_args()
    with io.open(args.config_path, encoding="utf8") as config_file:
        config = json.load(config_file)
    run_and_save_registry_metrics(**config)
