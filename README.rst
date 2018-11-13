Snips NLU Metrics
=================

.. image:: https://travis-ci.org/snipsco/snips-nlu-metrics.svg?branch=master
    :target: https://travis-ci.org/snipsco/snips-nlu-metrics

.. image:: https://img.shields.io/pypi/v/snips-nlu-metrics.svg?branch=master
    :target: https://pypi.python.org/pypi/snips-nlu-metrics

.. image:: https://img.shields.io/pypi/pyversions/snips-nlu-metrics.svg?branch=master
    :target: https://pypi.python.org/pypi/snips-nlu-metrics


This tools is a python library for computing `cross-validation`_ and
`train/test`_ metrics on an NLU parsing pipeline such as the `Snips NLU`_ one.

Its purpose is to help evaluating and iterating on the tested intent parsing
pipeline.

Install
-------

.. code-block:: console

    $ pip install snips_nlu_metrics


NLU Metrics API
---------------

Snips NLU metrics API consists in the following functions:

* ``compute_train_test_metrics`` to compute `train/test`_ metrics
* ``compute_cross_val_metrics`` to compute `cross-validation`_ metrics

The metrics output (json) provides detailed information about:

* `confusion matrix`_
* `precision, recall and f1 scores`_ of intent classification
* precision, recall and f1 scores of entity extraction
* parsing errors

Data
----

Some sample datasets, that can be used to compute metrics, are available
`here <samples/>`_. Alternatively, you can create your own dataset either by
using ``snips-nlu``'s `dataset generation tool`_ or by going on the
`Snips console`_.

Examples
--------

The Snips NLU metrics library can be used with any NLU pipeline which satisfies
the ``Engine`` API:

.. code-block:: python

    from builtins import object

    class Engine(object):
        def fit(self, dataset):
            # Perform training ...
            return self

        def parse(self, text):
            # extract intent and slots ...
            return {
                "input": text,
                "intent": {
                    "intentName": intent_name,
                    "probability": probability
                },
                "slots": slots
            }


----------------
Snips NLU Engine
----------------

This library can be used to benchmark NLU solutions such as `Snips NLU`_. To
install the ``snips-nlu`` python library, and fetch the language resources for
english, run the following commands:

.. code-block:: bash

    $ pip install snips-nlu
    $ snips-nlu download en


Then, you can compute metrics for the ``snips-nlu`` pipeline using the metrics
API as follows:

.. code-block:: python

    from snips_nlu import load_resources, SnipsNLUEngine
    from snips_nlu_metrics import compute_train_test_metrics, compute_cross_val_metrics

    load_resources("en")

    tt_metrics = compute_train_test_metrics(train_dataset="samples/train_dataset.json",
                                            test_dataset="samples/test_dataset.json",
                                            engine_class=SnipsNLUEngine)

    cv_metrics = compute_cross_val_metrics(dataset="samples/cross_val_dataset.json",
                                           engine_class=SnipsNLUEngine,
                                           nb_folds=5)

-----------------
Custom NLU Engine
-----------------

You can also compute metrics on a custom NLU engine, here is a simple example:

.. code-block:: python

    import random

    from snips_nlu_metrics import compute_train_test_metrics

    class MyNLUEngine(object):
        def fit(self, dataset):
            self.intent_list = list(dataset["intents"])
            return self

        def parse(self, text):
            return {
                "input": text,
                "intent": {
                    "intentName": random.choice(self.intent_list),
                    "probability": 0.5
                },
                "slots": []
            }

    compute_train_test_metrics(train_dataset="samples/train_dataset.json",
                               test_dataset="samples/test_dataset.json",
                               engine_class=MyNLUEngine)

Links
-----
* `Changelog <CHANGELOG.md>`_
* `Bug tracker <https://github.com/snipsco/snips-nlu-metrics/issues>`_
* `Snips NLU <https://github.com/snipsco/snips-nlu>`_
* `Snips NLU Rust <https://github.com/snipsco/snips-nlu-rs>`_: Rust inference pipeline implementation and bindings (C, Swift, Kotlin, Python)
* `Snips <https://snips.ai/>`_

Contributing
------------
Please see the `Contribution Guidelines <CONTRIBUTING.rst>`_.

Copyright
---------
This library is provided by `Snips <https://www.snips.ai>`_ as Open Source software. See `LICENSE <LICENSE>`_ for more information.

.. _cross-validation: https://en.wikipedia.org/wiki/Cross-validation_(statistics)
.. _train/test: https://en.wikipedia.org/wiki/Training,_test,_and_validation_sets
.. _Snips NLU: https://github.com/snipsco/snips-nlu
.. _precision, recall and f1 scores: https://en.wikipedia.org/wiki/Precision_and_recall
.. _confusion matrix: https://en.wikipedia.org/wiki/Confusion_matrix
.. _dataset generation tool: http://snips-nlu.readthedocs.io/en/latest/tutorial.html#snips-dataset-format
.. _Snips console: https://console.snips.ai