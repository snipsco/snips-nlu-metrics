import io
import json

from nlu_metrics.metrics import run_registry_metrics

with io.open("sample_metrics_config.json") as f:
    config = json.load(f)

run_registry_metrics(config)
