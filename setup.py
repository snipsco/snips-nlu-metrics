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

required = [
    "pytest",
    "mock==2.0.0",
]

setup(name=PACKAGE_NAME,
      version=version,
      author="Adrien Ball",
      author_email="adrien.ball@snips.ai",
      license="All rights reserved",
      install_requires=required,
      packages=packages,
      package_data={
          "": [
              VERSION,
              "tests/resources/*"
          ]},
      entry_points={},
      include_package_data=True,
      zip_safe=False)
