from snips_nlu_metrics.utils.constants import UTTERANCES, INTENTS


class NotEnoughDataError(Exception):
    def __init__(self, dataset, nb_folds, train_size_ratio):
        self.dataset = dataset
        self.nb_folds = nb_folds
        self.train_size_ratio = train_size_ratio
        self.intent_utterances = {
            intent: len(data[UTTERANCES]) for intent, data in dataset[INTENTS].items()
        }

    @property
    def message(self):
        return "Not enough data: %r" % self

    def __repr__(self):
        return ", ".join(
            [
                "nb folds = %s" % self.nb_folds,
                "train size ratio = %s" % self.train_size_ratio,
                "intents details = [%s]"
                % ", ".join(
                    "%s -> %d utterances" % (intent, nb)
                    for intent, nb in sorted(self.intent_utterances.items())
                ),
            ]
        )

    def __str__(self):
        return repr(self)
