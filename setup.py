from pathlib import Path

from setuptools import setup, find_packages

packages = [p for p in find_packages() if "tests" not in p]

PACKAGE_NAME = "snips_nlu_metrics"
ROOT_PATH = Path(__file__).resolve().parent
PACKAGE_PATH = ROOT_PATH / PACKAGE_NAME
README = ROOT_PATH / "README.rst"
VERSION = "__version__"

with (PACKAGE_PATH / VERSION).open() as f:
    version = f.readline().strip()

with README.open(encoding="utf8") as f:
    readme = f.read()

install_requires = [
    "numpy>=1.7,<2.0",
    "scipy>=1.0,<2.0",
    "scikit-learn>=0.21.0,<0.23; python_version>='3.5'",
    "joblib>=0.13,<0.15",
]

extras_require = {"test": ["mock>=2.0,<3.0", "pytest>=5.3.1,<6",]}

setup(
    name=PACKAGE_NAME,
    description="Python package to compute NLU metrics",
    long_description=readme,
    version=version,
    author="Adrien Ball",
    author_email="adrien.ball@snips.ai",
    license="Apache 2.0",
    url="https://github.com/snipsco/snips-nlu-metrics",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="metrics nlu nlp intent slots entity parsing",
    extras_require=extras_require,
    install_requires=install_requires,
    packages=packages,
    include_package_data=True,
    zip_safe=False,
)
