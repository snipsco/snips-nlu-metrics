import shutil
from builtins import object
from tempfile import mkdtemp


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
