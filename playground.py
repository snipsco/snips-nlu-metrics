from nlu_metrics import compute_cross_val_metrics
from snips_nlu import SnipsNLUEngine
from snips_nlu_rust import NLUEngine as RustNLUEngine


class EndToEndTrainingEngine(object):
    def __init__(self, language):
        self.language = language
        self.nlu_engine = SnipsNLUEngine(language)

    def fit(self, dataset):
        self.nlu_engine.fit(dataset)
        pass

    def to_dict(self):
        return dict()

