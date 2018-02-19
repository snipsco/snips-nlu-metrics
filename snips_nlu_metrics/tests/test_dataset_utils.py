from __future__ import unicode_literals

import unittest

from snips_nlu_metrics.utils.dataset_utils import get_utterances_subset, \
    update_entities_with_utterances


class TestDatasetUtils(unittest.TestCase):
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
        utterances_subset = sorted(
            utterances_subset,
            key=lambda u: "%s%s" % (u[0], u[1]["data"][0]["text"]))

        # Then
        expected_utterances = [
            ("intent1", {"data": [{"text": "text1"}]}),
            ("intent1", {"data": [{"text": "text2"}]}),
            ("intent2", {"data": [{"text": "text1"}]}),
            ("intent3", {"data": [{"text": "text1"}]}),
            ("intent3", {"data": [{"text": "text2"}]}),
            ("intent3", {"data": [{"text": "text3"}]}),
        ]
        self.assertListEqual(expected_utterances, utterances_subset)

    def test_update_entities_with_utterances(self):
        # Given
        dataset = {
            "intents": {
                "intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "aa",
                                    "entity": "entity_2"
                                },
                                {
                                    "text": "bb",
                                    "entity": "entity_2"
                                }
                            ]
                        }
                    ]
                },
                "intent_2": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "cccc",
                                    "entity": "entity_1"
                                }
                            ]
                        }
                    ]
                },

            },
            "entities": {
                "entity_1": {
                    "data": [],
                    "use_synonyms": False
                },
                "entity_2": {
                    "data": [
                        {
                            "value": "a",
                            "synonyms": ["aa"]
                        }
                    ],
                    "use_synonyms": True
                }
            }
        }
        # When
        updated_dataset = update_entities_with_utterances(dataset)

        # Then
        expected_dataset = {
            "intents": {
                "intent_1": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "aa",
                                    "entity": "entity_2"
                                },
                                {
                                    "text": "bb",
                                    "entity": "entity_2"
                                }
                            ]
                        }
                    ]
                },
                "intent_2": {
                    "utterances": [
                        {
                            "data": [
                                {
                                    "text": "cccc",
                                    "entity": "entity_1"
                                }
                            ]
                        }
                    ]
                },

            },
            "entities": {
                "entity_1": {
                    "data": [
                        {
                            "value": "cccc",
                            "synonyms": []
                        }
                    ],
                    "use_synonyms": False
                },
                "entity_2": {
                    "data": [
                        {
                            "value": "a",
                            "synonyms": ["aa"]
                        },
                        {
                            "value": "bb",
                            "synonyms": []
                        }
                    ],
                    "use_synonyms": True
                }
            }
        }
        self.assertDictEqual(expected_dataset, updated_dataset)
