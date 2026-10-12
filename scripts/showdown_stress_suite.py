"""Validate declarative Showdown stress themes without generating battles."""
import json
from pathlib import Path


def load_suite(path, pool_path):
    suite = json.loads(Path(path).read_text())
    pool = json.loads(Path(pool_path).read_text())
    if suite.get('schema_version') != 1 or not isinstance(suite.get('themes'), list) or not suite['themes']:
        raise ValueError('stress suite needs schema_version 1 and a nonempty themes list')
    known = {
        '--moves': {move for species in pool['species'] for move in species['moves']},
        '--abilities': set(pool['abilities']) | {'noability'},
        '--items': set(pool['items']) | set(pool.get('megastones', [])),
        '--species': {species['id'] for species in pool['species']},
    }
    themes, names = [], set()
    for theme in suite['themes']:
        name, options = theme.get('name'), theme.get('options')
        if not isinstance(name, str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name) or name in names:
            raise ValueError(f'invalid or duplicate theme name: {name!r}')
        names.add(name)
        if not isinstance(options, list) or len(options) % 2 or not all(isinstance(s, str) for s in options):
            raise ValueError(f'{name}: options must be flag/value string pairs')
        flags = set()
        for flag, value in zip(options[::2], options[1::2]):
            if flag in flags:
                raise ValueError(f'{name}: duplicate {flag}')
            flags.add(flag)
            if flag in known:
                values = value.split(',')
                unknown = set(values) - known[flag]
                if unknown:
                    raise ValueError(f'{name}: unsupported {flag}: {sorted(unknown)}')
            elif flag == '--moves-per':
                if value not in {'1', '2', '3', '4'}:
                    raise ValueError(f'{name}: moves-per must be 1..4')
            else:
                raise ValueError(f'{name}: unsupported option {flag}')
        themes.append((name, options))
    rates = suite.get('mega_rates', [])
    if not isinstance(rates, list) or any(type(r) not in (int, float) or not 0 <= r <= 1 for r in rates):
        raise ValueError('mega_rates must be a list of probabilities in 0..1')
    return themes, rates
