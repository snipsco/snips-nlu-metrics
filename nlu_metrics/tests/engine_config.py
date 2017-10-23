NLU_CONFIG = {
    'intent_classifier_config': {
        'data_augmentation_config': {
            'min_utterances': 20,
            'noise_factor': 5
        },
        'featurizer_config': {
            'sublinear_tf': False
        },
        'log_reg_args': {
            u'class_weight': u'balanced',
            u'loss': u'log',
            u'n_iter': 5,
            u'n_jobs': -1,
            u'penalty': u'l2',
            u'random_state': 42
        }
    },
    'probabilistic_intent_parser_config': {
        'crf_features_config': {
            'base_drop_ratio': 0.0,
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
