from __future__ import unicode_literals

import io
import json
import os
import shutil
import zipfile
from tempfile import mkdtemp

TRAINED_ENGINE_FILENAME = "trained_assistant.json"


class tempdir_ctx(object):
    def __init__(self, suffix="", prefix="tmp", dir=None):
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir

    def __enter__(self):
        self.engine_dir = mkdtemp(suffix=self.suffix, prefix=self.prefix,
                                  dir=self.dir)
        return self.engine_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.engine_dir)


def get_trained_engine(train_dataset, training_engine_class):
    language = train_dataset["language"]
    engine = training_engine_class(language)
    engine.fit(train_dataset)
    return engine


def get_inference_engine(language, trained_engine_dict,
                         inference_engine_class):
    with tempdir_ctx() as engine_dir:
        trained_engine_path = os.path.join(engine_dir, TRAINED_ENGINE_FILENAME)
        archive_path = os.path.join(engine_dir, 'assistant.zip')

        with io.open(trained_engine_path, mode='w', encoding='utf8') as f:
            f.write(json.dumps(trained_engine_dict).decode())
        with zipfile.ZipFile(archive_path, 'w') as zf:
            zf.write(trained_engine_path, arcname=TRAINED_ENGINE_FILENAME)
        with io.open(archive_path, mode='rb') as f:
            data_zip = bytearray(f.read())

    return inference_engine_class(language, data_zip=data_zip)
