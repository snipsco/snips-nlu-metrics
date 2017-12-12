from __future__ import unicode_literals

NLU_CONFIG = {
    'intent_classifier_config': {
        'data_augmentation_config': {
            'min_utterances': 20,
            'noise_factor': 5,
            'unknown_word_prob': 0.2,
            'unknown_words_replacement_string': 'unknownword'
        },
        'featurizer_config': {
            'sublinear_tf': False
        },
        'log_reg_args': {
            'class_weight': 'balanced',
            'loss': 'log',
            'n_iter': 5,
            'n_jobs': -1,
            'penalty': 'l2',
        }
    },
    'probabilistic_intent_parser_config': {
        'crf_features_config': {
            'features_drop_out': {
                'collection_match': 0.1
            },
            'entities_offsets': [-2, -1, 0]
        },
        'data_augmentation_config': {
            'capitalization_ratio': 0.0,
            'min_utterances': 200
        }
    },
    'regex_training_config': {
        'max_entities': 200,
        'max_queries': 50
    }
}
