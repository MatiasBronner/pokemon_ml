#!/usr/bin/env python3
"""Checks scrape_teams.py and teampool against a made-up tournament, without a network.

    scripts/check_scraper.py [teams]

Asks the engine for random legal teams (`teamcheck --sample`), writes a
tournament page and one paste page per team into a cache the way a browser
would have saved them, runs the scraper on that cache and `teampool` on its
output, and requires every team to come back as it went in.

The paste pages are laid out five different ways, because the real site is
script-driven and how it delivers a sheet can change: as plain text, as
separate elements with no punctuation, inside a script as text, inside a
script as data, and as text with a copy in a script.
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


def export(team, natures):
    """A team in Showdown's export format, as an open team sheet has it: no stat points."""
    blocks = []
    for mon in team:
        lines = [mon['species'] + (' @ ' + mon['item'] if mon['item'] else ''), 'Ability: ' + mon['ability'], 'Level: 50']
        if natures:
            lines.append(mon['nature'] + ' Nature')
        lines += ['- ' + m for m in mon['moves']]
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks)


def shell(title, body, scripts=''):
    return (f'<!DOCTYPE html><html><head><title>{html.escape(title)}</title><style>.x{{color:red}}</style></head>'
            f'<body><nav><a href="/">VR Pastes - Victory Road</a><span>Beta</span></nav>{body}{scripts}</body></html>')


def flight(text):
    """Text the way a Next.js page carries it in a script: quoted, inside a quoted chunk."""
    return '<script>self.__next_f.push([1,' + json.dumps('7:["$","div",null,{"paste":' + json.dumps(text) + '}]\n') + '])</script>'


def paste_page(layout, title, team, natures):
    text = export(team, natures)
    if layout == 0:  # plain text, names wrapped in spans
        marked = html.escape(text).replace(' @ ', ' @ <span class="item">').replace('\n', '</span>\n')
        return shell(title, f'<main><pre>{marked}</pre></main>')
    if layout == 1:  # one element per fact, streamed into a hidden block
        cards = ''
        for mon in team:
            cards += f'<article><img alt="{html.escape(mon["species"])}" src="/s.png"><h3>{html.escape(mon["species"])}</h3>'
            cards += f'<p class="item">{html.escape(mon["item"])}</p><p>{html.escape(mon["ability"])}</p>'
            cards += (f'<p>{mon["nature"]}</p>' if natures else '')
            cards += '<ul>' + ''.join(f'<li>{html.escape(m)}</li>' for m in mon['moves']) + '</ul></article>'
        return shell(title, f'<main><template id="B:0"></template></main><div hidden id="S:0">{cards}</div>')
    if layout == 2:  # only inside a script, as text
        return shell(title, '<main></main>', flight(text))
    if layout == 3:  # only inside a script, as data
        data = [{'species': m['species'], 'item': m['item'], 'ability': m['ability'], 'moves': m['moves'],
                 **({'nature': m['nature']} if natures else {})} for m in team]
        chunk = '7:["$","div",null,{"team":' + json.dumps(data) + '}]\n'
        return shell(title, '<main></main>', '<script>self.__next_f.push([1,' + json.dumps(chunk) + '])</script>')
    # text on the page and again in a script, after a strip of the six Pokémon
    strip = ''.join(f'<img alt="{html.escape(m["species"])}"><span>{html.escape(m["species"])}</span>' for m in team)
    return shell(title, f'<header>{strip}</header><main><pre>{html.escape(text)}</pre></main>', flight(text))


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
            title = f"{player}'s 2099 Nowhere Regional Championships OTS – VR Pastes"
            page = paste_page(i % 5, title, team, natures=i % 3 != 0)
            with open(scrape_teams.cache_path('https://www.vrpastes.com/' + pid, cache), 'w', encoding='utf-8') as f:
                f.write(page)
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
                                  '--cache', cache, '--delay', '0'], capture_output=True, text=True)
        print(scraped.stdout.strip())
        assert scraped.returncode == 0, scraped.stderr
        out = os.path.join(tmp, 'nowhere.json')
        pooled = subprocess.run([os.path.join(REPO, 'target', 'release', 'teampool'), raw, '--out', out, '--play', '200'],
                                capture_output=True, text=True)
        print(pooled.stdout.strip())
        assert pooled.returncode == 0, pooled.stderr
        pool = json.load(open(out))
        rows_read = json.load(open(raw))['teams']
        assert len(rows_read) == n, f'{len(rows_read)} rows read from the tournament page, not {n}'
        assert [r['placing'] for r in rows_read] == list(range(1, n + 1)), 'placings'
        assert rows_read[0]['country'] == 'ESP' and rows_read[3]['record'] == '10-3', rows_read[3]
        assert rows_read[-1]['section'] == 'Senior Division' and rows_read[0]['section'] == 'Top Cut'
        assert len(pool['teams']) == n, f'{len(pool["teams"])} of {n} teams came back'
        for i, (sent, got) in enumerate(zip(teams, pool['teams'])):
            assert got['player'] == (f"Player O'Num{i}" if i % 7 == 0 else f'Player Num{i}'), got['player']
            assert got['placing'] == i + 1 and got['spreads_guessed']
            assert got['natures_guessed'] == (i % 3 == 0)
            for a, b in zip(sent, got['team']):
                same = (a['species'], a['item'], a['ability'], a['moves']) == (b['species'], b['item'], b['ability'], b['moves'])
                assert same, f'team {i} (layout {i % 5}): sent {a}, got {b}'
                assert i % 3 == 0 or a['nature'] == b['nature'] or {a['nature'], b['nature']} <= {
                    'Hardy', 'Docile', 'Serious', 'Bashful', 'Quirky'}, (a['nature'], b['nature'])
                assert sum(b['evs'].values()) == 66
    print(f'all {n} teams came back as they went in, from five page layouts')


if __name__ == '__main__':
    main()
