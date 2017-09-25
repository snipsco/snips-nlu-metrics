from __future__ import unicode_literals

import unittest

from nlu_metrics.utils.dataset_utils import get_stratified_utterances, \
    get_utterances_subset


class TestDatasetUtils(unittest.TestCase):
    def test_stratified_utterances_should_work(self):
        # Given
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
        utterances = get_stratified_utterances(dataset, seed=42)

        # Then
        expected_utterances = [
            ('intent1', {'data': [{'text': 'text3'}]}),
            ('intent3', {'data': [{'text': 'text3'}]}),
            ('intent2', {'data': [{'text': 'text1'}]}),
            ('intent1', {'data': [{'text': 'text1'}]}),
            ('intent3', {'data': [{'text': 'text4'}]}),
            ('intent2', {'data': [{'text': 'text2'}]}),
            ('intent1', {'data': [{'text': 'text2'}]}),
            ('intent3', {'data': [{'text': 'text1'}]}),
            ('intent3', {'data': [{'text': 'text2'}]})
        ]
        self.assertListEqual(expected_utterances, utterances)

    def test_get_utterances_subset_should_work(self):
        # Given
        utterances = [
            ('intent1', {'data': [{'text': 'text1'}]}),
            ('intent1', {'data': [{'text': 'text2'}]}),
            ('intent1', {'data': [{'text': 'text3'}]}),
            ('intent1', {'data': [{'text': 'text4'}]}),
            ('intent2', {'data': [{'text': 'text1'}]}),
            ('intent2', {'data': [{'text': 'text2'}]}),
            ('intent3', {'data': [{'text': 'text1'}]}),
            ('intent3', {'data': [{'text': 'text2'}]}),
            ('intent3', {'data': [{'text': 'text3'}]}),
            ('intent3', {'data': [{'text': 'text4'}]}),
            ('intent3', {'data': [{'text': 'text5'}]}),
            ('intent3', {'data': [{'text': 'text6'}]}),
        ]

        # When
        utterances_subset = get_utterances_subset(utterances, ratio=0.5)

        # Then
        expected_utterances = [
            ("intent1", {"data": [{"text": "text1"}]}),
            ("intent1", {"data": [{"text": "text2"}]}),
            ("intent2", {"data": [{"text": "text1"}]}),
            ("intent3", {"data": [{"text": "text1"}]}),
            ("intent3", {"data": [{"text": "text2"}]}),
            ("intent3", {"data": [{"text": "text3"}]}),
        ]
        self.assertItemsEqual(expected_utterances, utterances_subset)
