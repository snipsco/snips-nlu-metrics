import json
import logging
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def resources_path():
    return Path(__file__).resolve().parent / "resources"


@pytest.fixture(scope="session")
def beverage_dataset_path(resources_path):
    return resources_path / "beverage_dataset.json"


@pytest.fixture(scope="session")
def beverage_dataset(resources_path):
    with (resources_path / "beverage_dataset.json").open(encoding="utf8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def keyword_matching_dataset(resources_path):
    with (resources_path / "keyword_matching_dataset.json").open(encoding="utf8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def logger():
    logger = logging.getLogger("snips_nlu_metrics")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
