# coding=utf-8
from __future__ import unicode_literals

import shutil
from subprocess import check_call

import pytest
from future.builtins import str
from pathlib import Path

from snips_nlu_metrics.tests import TEST_PATH

try:
    from tempfile import TemporaryDirectory
except ImportError:
    from backports.tempfile import TemporaryDirectory


@pytest.fixture()
def tmp_dir(request):
    with TemporaryDirectory() as dir:
        yield dir


def test_train_test_split_cli(tmp_dir):
    # Given
    tmp_dir = Path(tmp_dir)

    samples_path = TEST_PATH.parent.parent.parent / "samples"
    sample_dataset_path = samples_path / "train_dataset.json"

    dataset_path = tmp_dir / "dataset.json"

    shutil.copy(str(sample_dataset_path), str(dataset_path))

    cmd = ["snips-nlu-metrics", "train-test-split", str(dataset_path),
           str(0.2), "-d", "-s", str(666)]

    # When / Then
    try:
        check_call(cmd)
    except Exception as e:
        pytest.fail("Exception occured while splitting the dataset: %s" % e)
