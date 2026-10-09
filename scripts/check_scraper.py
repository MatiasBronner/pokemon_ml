#!/usr/bin/env python3
"""Checks scrape_teams.py and teampool against a made-up tournament, without a network.

    scripts/check_scraper.py [teams]

Asks the engine for random legal teams (`teamcheck --sample`), writes a
tournament page and, for each team, the answer vrpastes.com's data service
gives for a paste, into a cache as the scraper would have kept them. Then it
runs the scraper on that cache and `teampool` on its output, and requires
every team to come back as it went in.

The answers have the shape of the real ones (checked against the service in
October 2026): a title and a list of Pokémon with species, item, ability,
moves and nature among much else. Some are given without natures, some with stat
points, and two cannot be had at all, which must be left out and said so.
"""
import html
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import scrape_teams  # noqa: E402

# Nothing listens here: a paste missing from the cache is refused at once, without a network.
BACKEND = 'http://127.0.0.1:9'


def shell(title, body):
    return (f'<!DOCTYPE html><html><head><title>{html.escape(title)}</title><style>.x{{color:red}}</style></head>'
            f'<body><nav><a href="/">Victory Road</a></nav>{body}</body></html>')


def answer(pid, title, team, natures, points):
    """What the data service says for a paste."""
    mons = []
    for mon in team:
        entry = {'name': mon['species'], 'species': mon['species'], 'item': mon['item'], 'ability': mon['ability'],
                 'moves': mon['moves'], 'type1': 'Normal', 'image': mon['species'].lower(),
                 'baseStats': {'hp': 80, 'atk': 80, 'def': 80, 'spa': 80, 'spd': 80, 'spe': 80},
                 'movesWithTypes': [{'name': m, 'type': 'Normal', 'translation': m} for m in mon['moves']],
                 'speciesTranslation': mon['species'], 'itemTranslation': mon['item'], 'abilityTranslation': mon['ability']}
        if natures:
            entry.update(nature=mon['nature'], natureTranslation=mon['nature'])
        if points:
            entry['evs'] = mon['evs']
        mons.append(entry)
    return {'id': pid, 'is_public': True, 'is_encrypted': False, 'title': title, 'format': 'VGC Regulation Set M-C',
            'teams': mons, 'hasPassword': False, 'createdAt': 1790575151}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    sample = subprocess.run([os.path.join(REPO, 'target', 'release', 'teamcheck'), '--sample', str(n), '--seed', '77'],
                            capture_output=True, text=True, check=True).stdout
    teams = [json.loads(line) for line in sample.splitlines() if line.strip()]
    assert len(teams) == n, 'build the engine first: cargo build --release'
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, 'cache')
        os.makedirs(cache)
        url = 'https://victoryroad.pro/2099-nowhere/'
        rows = {'Top Cut': '', 'Swiss rounds': '', 'Senior Division': ''}
        for i, team in enumerate(teams):
            pid = f'Fake{i:04d}'
            host = 'https://www.vrpastes.com/' if i % 2 else 'https://vrpastes.com/'
            player = f"Player O'Num{i}" if i % 7 == 0 else f'Player Num{i}'
            title = f"{player}'s 2099 Nowhere Regional Championships OTS"
            kept = scrape_teams.cache_path(f'{BACKEND}/api/paste/{pid}?lang=english', cache)
            with open(kept, 'w', encoding='utf-8') as f:
                # Two pastes with no team kept for them: one an answer that is no team, one something else
                # altogether. The scraper asks again for both, and here gets no answer.
                if i == n - 4:
                    json.dump({'error': 'Password required', 'hasPassword': True}, f)
                elif i == n - 5:
                    f.write('<html><body>Not Found</body></html>')
                else:
                    json.dump(answer(pid, title, team, natures=i % 3 != 0, points=i % 9 == 1), f)
            sprites = ''.join('<img src="https://victoryroad.pro/wp-content/uploads/sprites/gen9-champions/'
                              + m['species'].lower().replace(' ', '-').replace("'", '').replace('.', '') + '.png">' for m in team)
            section = 'Senior Division' if i >= n - 3 else 'Top Cut' if i < 8 else 'Swiss rounds'
            rows[section] += (f'<tr><td>{i + 1}</td><td>{13 - i % 5}-{i % 5}</td>'
                              f'<td><img src="https://victoryroad.pro/wp-content/uploads/flags/ESP-flag.png"></td>'
                              f'<td><a href="https://x.com/p{i}">{html.escape(player)}</a></td><td>$1,000, 280 CP</td>'
                              f'<td>{sprites}</td><td><a href="{host}{pid}"><img title="Export Team" src="/e.png"></a></td></tr>\n')
        body = ''.join(f'<h3>{name}</h3><table><tr><th>#</th><th>Swiss</th><th>Player</th></tr>{trs}</table>'
                       for name, trs in rows.items())
        with open(scrape_teams.cache_path(url, cache), 'w', encoding='utf-8') as f:
            f.write(shell('2099 Nowhere Regional Championships', f'<article><h1>2099 Nowhere</h1>{body}</article>'))

        raw = os.path.join(tmp, 'raw', 'nowhere.json')
        scraped = subprocess.run([sys.executable, os.path.join(REPO, 'scripts', 'scrape_teams.py'), url, '--out', raw,
                                  '--cache', cache, '--delay', '0', '--backend', BACKEND], capture_output=True, text=True)
        print(scraped.stdout.strip())
        assert scraped.returncode == 0, scraped.stderr
        out = os.path.join(tmp, 'nowhere.json')
        pooled = subprocess.run([os.path.join(REPO, 'target', 'release', 'teampool'), raw, '--out', out, '--play', '2000', '--vary'],
                                capture_output=True, text=True)
        print(pooled.stdout.strip())
        assert pooled.returncode == 0, pooled.stderr
        pool = json.load(open(out))
        rows_read = json.load(open(raw))['teams']
        assert len(rows_read) == n, f'{len(rows_read)} rows read from the tournament page, not {n}'
        assert [r['placing'] for r in rows_read] == list(range(1, n + 1)), 'placings'
        assert rows_read[0]['country'] == 'ESP' and rows_read[3]['record'] == '10-3', rows_read[3]
        assert rows_read[-1]['section'] == 'Senior Division' and rows_read[0]['section'] == 'Top Cut'
        gone = [n - 5, n - 4]
        assert [i for i, r in enumerate(rows_read) if 'error' in r] == gone, 'the two pastes that cannot be had'
        assert f'{n - 2} of {n} sheets list six' in scraped.stdout
        assert len(pool['teams']) == n - 2, f'{len(pool["teams"])} of {n - 2} teams came back'
        for got in pool['teams']:
            i = got['placing'] - 1
            sent = teams[i]
            assert i not in gone
            assert got['player'] == (f"Player O'Num{i}" if i % 7 == 0 else f'Player Num{i}'), got['player']
            assert got['spreads_guessed'] == (i % 9 != 1) and got['natures_guessed'] == (i % 3 == 0)
            for a, b in zip(sent, got['team']):
                same = (a['species'], a['item'], a['ability'], a['moves']) == (b['species'], b['item'], b['ability'], b['moves'])
                assert same, f'team {i}: sent {a}, got {b}'
                assert i % 3 == 0 or a['nature'] == b['nature'] or {a['nature'], b['nature']} <= {
                    'Hardy', 'Docile', 'Serious', 'Bashful', 'Quirky'}, (a['nature'], b['nature'])
                assert sum(b['evs'].values()) == 66
                assert i % 9 != 1 or a['evs'] == b['evs'], f'team {i}: stat points sent {a["evs"]}, got {b["evs"]}'

        # A site that answers with no teams at all must stop both steps, not leave an empty pool behind.
        empty = os.path.join(tmp, 'empty')
        os.makedirs(os.path.join(empty, 'cache'))
        with open(scrape_teams.cache_path(url, os.path.join(empty, 'cache')), 'w', encoding='utf-8') as f:
            f.write(shell('2099 Nowhere', f'<article>{body}</article>'))
        raw = os.path.join(empty, 'raw.json')
        scraped = subprocess.run([sys.executable, os.path.join(REPO, 'scripts', 'scrape_teams.py'), url, '--out', raw,
                                  '--cache', os.path.join(empty, 'cache'), '--delay', '0', '--backend', BACKEND, '--top', '3'],
                                 capture_output=True, text=True)
        assert scraped.returncode != 0 and 'No team came back' in scraped.stderr, scraped.stderr
        out = os.path.join(empty, 'pool.json')
        pooled = subprocess.run([os.path.join(REPO, 'target', 'release', 'teampool'), raw, '--out', out],
                                capture_output=True, text=True)
        assert pooled.returncode != 0 and not os.path.exists(out), pooled.stdout
    print(f'all {n - 2} teams came back as they went in; the two that could not be had, and an empty answer, were said so')


if __name__ == '__main__':
    main()
