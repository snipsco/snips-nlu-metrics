import io
import os

from setuptools import setup, find_packages

packages = [p for p in find_packages() if "tests" not in p]

PACKAGE_NAME = "snips_nlu_metrics"
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.join(ROOT_PATH, PACKAGE_NAME)
README = os.path.join(ROOT_PATH, "README.rst")
VERSION = "__version__"

with io.open(os.path.join(PACKAGE_PATH, VERSION)) as f:
    version = f.readline().strip()

with io.open(README, 'rt', encoding='utf8') as f:
    readme = f.read()

install_requires = [
    "future",
    "numpy>=1.7,<2.0",
    "scipy>=1.0,<2.0",
    "scikit-learn>=0.19,<0.20",
    "pathos~=0.2"
]

extras_require = {
    "test": [
        "mock>=2.0,<3.0",
    ]
}

setup(name=PACKAGE_NAME,
      description="Python package to compute NLU metrics",
      long_description=readme,
      version=version,
      author="Adrien Ball",
      author_email="adrien.ball@snips.ai",
      license="Apache 2.0",
      classifiers=[
          "Programming Language :: Python :: 2",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
      ],
      extras_require=extras_require,
      install_requires=install_requires,
      packages=packages,
      include_package_data=True,
      zip_safe=False)
