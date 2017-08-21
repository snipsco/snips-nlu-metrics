from __future__ import unicode_literals

import unittest

from snips_nlu import SnipsNLUEngine

from nlu_metrics.utils.nlu_engine_utils import get_trained_nlu_engine


class TestNLUEngineUtils(unittest.TestCase):
    def test_should_use_provided_training_class(self):
        # Given
        class TestNLUEngine(SnipsNLUEngine):
            fit_calls = 0

            def fit(self, dataset, intents=None):
                TestNLUEngine.fit_calls += 1
                super(TestNLUEngine, self).fit(dataset, intents)

        _dataset = {
            "language": "en",
            "intents": {
                "intent1": {
                    "engineType": "regex",
                    "utterances": [
                        {"data": [{"text": "text1"}]},
                        {"data": [{"text": "text2"}]},
                        {"data": [{"text": "text3"}]}
                    ]
                },
                "intent2": {
                    "engineType": "regex",
                    "utterances": [
                        {"data": [{"text": "text1"}]},
                        {"data": [{"text": "text2"}]},
                    ]
                },
                "intent3": {
                    "engineType": "regex",
                    "utterances": [
                        {"data": [{"text": "text1"}]},
                        {"data": [{"text": "text2"}]},
                        {"data": [{"text": "text3"}]},
                        {"data": [{"text": "text4"}]}
                    ]
                },
            },
            "entities": {},
            "snips_nlu_version": "0.1.0"
        }

        # When
        get_trained_nlu_engine(_dataset, TestNLUEngine)

        # Then
        self.assertGreater(TestNLUEngine.fit_calls, 0)
