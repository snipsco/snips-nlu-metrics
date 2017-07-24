from __future__ import unicode_literals

import io
import json
import os
import shutil
import zipfile
from tempfile import mkdtemp

from snips_nlu.nlu_engine import SnipsNLUEngine
from snips_nlu_rust import NLUEngine as RustNLUEngine

from nlu_metrics.utils.dependency_utils import update_nlu_packages
from nlu_metrics.utils.metrics_utils import create_k_fold_batches

TRAINED_ENGINE_FILENAME = "trained_assistant.json"


def compute_metrics(language, dataset, snips_nlu_version,
                    snips_nlu_rust_version):
    """
    Compute the main NLU metrics on the provided dataset and return a dict
    """
    update_nlu_packages(snips_nlu_version=snips_nlu_version,
                        snips_nlu_rust_version=snips_nlu_rust_version)

    batches = create_k_fold_batches(dataset, k=5)

    for (train_dataset, test_utterances) in batches:
        engine = get_trained_nlu_engine(language, train_dataset)


def get_trained_nlu_engine(language, dataset):
    trained_engine_dict = SnipsNLUEngine(language).fit(dataset).to_dict()
    engine_dir = mkdtemp()
    try:
        trained_engine_path = os.path.join(engine_dir, TRAINED_ENGINE_FILENAME)
        archive_path = os.path.join(engine_dir, 'assistant.zip')

        with io.open(trained_engine_path, mode='w', encoding='utf8') as f:
            f.write(json.dumps(trained_engine_dict).decode())
        with zipfile.ZipFile(archive_path, 'w') as zf:
            zf.write(trained_engine_path, arcname=TRAINED_ENGINE_FILENAME)
        with io.open(archive_path, mode='rb') as f:
            data_zip = bytearray(f.read())
    except Exception as e:
        raise Exception("Error while creating engine from zip archive: %s"
                        % e.message)
    finally:
        shutil.rmtree(engine_dir)
    return RustNLUEngine(language, data_zip=data_zip)
