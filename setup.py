import io
import os

from setuptools import setup, find_packages

packages = [p for p in find_packages() if "tests" not in p]

PACKAGE_NAME = "nlu_metrics"
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.join(ROOT_PATH, PACKAGE_NAME)
VERSION = "__version__"

with io.open(os.path.join(PACKAGE_PATH, VERSION)) as f:
    version = f.readline().strip()

install_requires = [
    "future",
    "numpy",
    "scipy",
    "scikit-learn",
]

extras_require = {
    "test": [
        "mock==2.0.0",
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
      include_package_data=True,
      zip_safe=False)
