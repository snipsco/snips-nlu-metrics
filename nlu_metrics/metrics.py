from __future__ import unicode_literals

from nlu_metrics.utils.dependency_utils import update_nlu_packages
from nlu_metrics.utils.metrics_utils import (create_k_fold_batches,
                                             compute_engine_metrics,
                                             aggregate_metrics,
                                             compute_precision_recall)
from nlu_metrics.utils.nlu_engine_utils import get_trained_nlu_engine


def compute_metrics(language, dataset, snips_nlu_version,
                    snips_nlu_rust_version, k_fold_size=5,
                    max_utterances=None):
    """
    Compute the main NLU metrics on the provided dataset
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

    for intent_metrics in global_metrics["intents"].values():
        prec_rec_metrics = compute_precision_recall(intent_metrics)
        intent_metrics.update(prec_rec_metrics)
    for slot_metrics in global_metrics["slots"].values():
        prec_rec_metrics = compute_precision_recall(slot_metrics)
        slot_metrics.update(prec_rec_metrics)

    return global_metrics
