from __future__ import unicode_literals

from nlu_metrics.utils.dataset_utils import get_stratified_utterances
from nlu_metrics.utils.dependency_utils import update_nlu_packages
from nlu_metrics.utils.metrics_utils import (create_k_fold_batches,
                                             compute_engine_metrics,
                                             aggregate_metrics,
                                             compute_precision_recall)
from nlu_metrics.utils.nlu_engine_utils import get_trained_nlu_engine


def compute_cross_val_metrics(language, dataset, snips_nlu_version,
                              snips_nlu_rust_version, k_fold_size=5,
                              max_utterances=None):
    """Compute the main NLU metrics on the dataset using cross validation

    :param language: str
    :param dataset: dict
    :param snips_nlu_version: str, semver
    :param snips_nlu_rust_version: str, semver
    :param k_fold_size: int, number of folds to use for cross validation
    :param max_utterances: int, max number of utterances to use for training
    :return: dict containing the metrics

    """
    update_nlu_packages(snips_nlu_version=snips_nlu_version,
                        snips_nlu_rust_version=snips_nlu_rust_version)

    batches = create_k_fold_batches(dataset, k=k_fold_size,
                                    max_training_utterances=max_utterances)

    global_metrics = {
        "intents": dict(),
        "slots": dict()
    }

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
    :param snips_nlu_version: str, semver
    :param snips_nlu_rust_version: str, semver
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
