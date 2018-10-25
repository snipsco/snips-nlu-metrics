# coding=utf-8
from __future__ import unicode_literals

import json

import plac
from pathlib import Path

from snips_nlu_metrics.utils import train_test_split as _train_test_split


@plac.annotations(
    dataset_path=("Path to the dataset to split", "positional", None, str),
    test_ratio=("Proportion of the dataset to keep as test set",
                "positional", None, float),
    drop_entities=(
            'Whether to drop entities values in the "entity" field of the'
            ' dataset', "flag", "d"),
    seed=("Random seed", "option", "s", int)
)
def train_test_split(dataset_path, test_ratio, drop_entities, seed=None):
    dataset_path = Path(dataset_path)

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not 0 < test_ratio < 1.0:
        raise ValueError("test_ration must be in ]0, 1[")
    train_size_ratio = 1 - test_ratio

    train_dataset, test_dataset = _train_test_split(
        dataset, train_size_ratio=train_size_ratio,
        drop_entities=drop_entities, seed=seed)

    name = dataset_path.stem
    ext = dataset_path.suffix

    train_path = dataset_path.with_name(name + "_train").with_suffix(ext)
    with train_path.open("w", encoding="utf-8") as f:
        dataset_as_str = json.dumps(
            train_dataset, sort_keys=True, indent=2).decode("utf-8")
        f.write(dataset_as_str)
    print("Wrote train dataset to '%s'" % train_path)

    test_path = dataset_path.with_name(name + "_test").with_suffix(ext)
    with test_path.open("w", encoding="utf-8") as f:
        dataset_as_str = json.dumps(
            test_dataset, sort_keys=True, indent=2).decode("utf-8")
        f.write(dataset_as_str)
    print("Wrote test dataset to '%s'" % test_path)
