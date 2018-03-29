import argparse
import json
import os
import sys

from snips_nlu import SnipsNLUEngine, load_resources

from snips_nlu_metrics import (compute_train_test_metrics,
                               compute_cross_val_metrics)

SAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATASET_PATH = os.path.join(SAMPLES_DIR, "train_dataset.json")
TEST_DATASET_PATH = os.path.join(SAMPLES_DIR, "test_dataset.json")
CROSS_VAL_DATASET_PATH = os.path.join(SAMPLES_DIR, "cross_val_dataset.json")


def compute_sample_train_test_metrics():
    load_resources("en")
    return compute_train_test_metrics(
        train_dataset=TRAIN_DATASET_PATH,
        test_dataset=TEST_DATASET_PATH,
        engine_class=SnipsNLUEngine)


def compute_sample_cross_val_metrics():
    load_resources("en")
    return compute_cross_val_metrics(dataset=CROSS_VAL_DATASET_PATH,
                                     engine_class=SnipsNLUEngine,
                                     nb_folds=5)


def main_metrics():
    parser = argparse.ArgumentParser(
        description="Compute sample metrics on the Snips NLU parsing pipeline")
    parser.add_argument("metrics_type", type=str,
                        choices=["train-test", "cross-val"],
                        metavar="metrics_type",
                        help="Type of metrics to compute")
    args = parser.parse_args(sys.argv[1:])
    if args.metrics_type == "train_test":
        metrics = compute_sample_train_test_metrics()
    else:
        metrics = compute_sample_cross_val_metrics()
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main_metrics()
