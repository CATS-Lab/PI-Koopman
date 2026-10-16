"""Shared prediction-grid configuration for execution and aggregation."""
import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs/config.json'


def cells(axis, dataset, path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    settings = config[axis]
    if settings['future_s'] != 5:
        raise ValueError('The prediction implementation requires a five-second future.')
    if axis == 'duration':
        datasets = settings['datasets'] if dataset == 'both' else [dataset]
        if any(ds not in settings['datasets'] for ds in datasets):
            raise ValueError('Dataset is not configured for the duration axis.')
        return [(ds, b, settings['rate_hz'], settings['origin_s'])
                for ds in datasets for b in settings['budgets_s']]
    if dataset != settings['dataset']:
        raise ValueError('The rate axis uses CATS-CF only.')
    return [(dataset, settings['budget_s'], rate, origin)
            for rate in settings['rates_hz'] for origin in settings['origins_s']]
