import io
import json

from nlu_metrics import compute_batch_metrics

with io.open("sample_metrics_config.json") as f:
    config = json.load(f)

compute_batch_metrics(config)
