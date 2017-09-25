import io
import os

from setuptools import setup, find_packages

packages = [p for p in find_packages() if "tests" not in p]

PACKAGE_NAME = "nlu_metrics"
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.join(ROOT_PATH, PACKAGE_NAME)
VERSION = "__version__"
DEFAULT_TRAINING_VERSION = os.path.join("tests",
                                        "__default_training_version__")
DEFAULT_INFERENCE_VERSION = os.path.join("tests",
                                         "__default_inference_version__")

with io.open(os.path.join(PACKAGE_PATH, VERSION)) as f:
    version = f.readline().strip()

with io.open(os.path.join(PACKAGE_PATH, DEFAULT_TRAINING_VERSION)) as f:
    training_version = f.readline().strip()

with io.open(os.path.join(PACKAGE_PATH, DEFAULT_INFERENCE_VERSION)) as f:
    inference_version = f.readline().strip()

install_requires = []

extras_require = {
    "test": [
        "mock==2.0.0",
        "snips_nlu==%s" % training_version,
        "snips_nlu_rust==%s" % inference_version,
    ]
}

setup(name=PACKAGE_NAME,
      version=version,
      author="Adrien Ball",
      author_email="adrien.ball@snips.ai",
      license="All rights reserved",
      extras_require=extras_require,
      install_requires=install_requires,
      packages=packages,
      package_data={
          "": [
              VERSION,
              DEFAULT_TRAINING_VERSION,
              DEFAULT_INFERENCE_VERSION
          ]},
      include_package_data=True,
      zip_safe=False)
