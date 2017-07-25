from __future__ import unicode_literals
import unittest

from mock import patch
from snips_nlu.constants import DATA, TEXT

from nlu_metrics.utils.dataset_utils import get_stratified_utterances


class TestDatasetUtils(unittest.TestCase):
    @patch('nlu_metrics.utils.dataset_utils.shuffle_dataset')
    def test_stratified_utterances_should_work(self, mocked_shuffle_dataset):
        # Given
        mocked_shuffle_dataset.side_effect = None
        dataset = {
            "intents": {
                "intent1": {
                    "utterances": [
                        {"data": [{"text": "text1"}]},
                        {"data": [{"text": "text2"}]},
                        {"data": [{"text": "text3"}]}
                    ]
                },
                "intent2": {
                    "utterances": [
                        {"data": [{"text": "text1"}]},
                        {"data": [{"text": "text2"}]},
                    ]
                },
                "intent3": {
                    "utterances": [
                        {"data": [{"text": "text1"}]},
                        {"data": [{"text": "text2"}]},
                        {"data": [{"text": "text3"}]},
                        {"data": [{"text": "text4"}]}
                    ]
                },
            }
        }

        # When
        utterances = get_stratified_utterances(dataset)
        utterances_texts = [u[DATA][0][TEXT] for (intent, u) in utterances]

        # Then
        expected_utterances_texts = [
            'text1',
            'text1',
            'text1',
            'text2',
            'text2',
            'text2',
            'text3',
            'text3',
            'text4'
        ]
        self.assertListEqual(expected_utterances_texts, utterances_texts)
        mocked_shuffle_dataset.assert_called_once()
