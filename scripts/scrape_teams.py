#!/usr/bin/env python3
"""Collects the teams of a tournament from Victory Road.

    scripts/scrape_teams.py https://victoryroad.pro/2027-frankfurt/ [--out teams/raw/2027-frankfurt.json]
                            [--top N] [--delay SECONDS] [--cache DIR] [--refresh] [--backend URL]

Reads the tournament page, follows every team-sheet link in its results tables
(the "Export Team" links to vrpastes.com) and writes one JSON file with, for
each team, where it placed, who played it and its sheet in Pokémon Showdown's
export format. `teampool` (cargo run --release --bin teampool) turns the file
into teams the engine can play, and says which it could not use.

A paste's page at vrpastes.com is an empty frame: the script on it asks the
site's data service for the team once it is open in a browser. This asks that
service directly (--backend, should its address ever change: it is the address
in the site's script next to "/api/paste/").

Pages are kept in --cache (default: teams/raw/cache), so running it again only
fetches what is missing; --refresh fetches everything anew. One request a
second by default: these are other people's servers.

Needs only Python 3 and a network connection.
"""
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

AGENT = 'Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0'
PASTE = re.compile(r'https?://(?:www\.)?vrpastes\.com/([A-Za-z0-9_-]{4,})')
# The rest of a paste's address is its id; these are the site's own pages.
NOT_A_PASTE = {'opengraph-image', 'twitter-image', 'logo_paste', 'logo_vr', '_next', 'api'}
# Where the script on a paste's page gets the team from.
BACKEND = 'https://vrpaste-backend.vercel.app'
STATS = [('hp', 'HP'), ('atk', 'Atk'), ('def', 'Def'), ('spa', 'SpA'), ('spd', 'SpD'), ('spe', 'Spe')]


def cache_path(url, cache):
    """Where the page at `url` is kept."""
    kind = '.json' if '/api/' in url else '.html'
    return os.path.join(cache, re.sub(r'[^A-Za-z0-9._-]+', '_', url.split('://', 1)[1]).strip('_') + kind)


def fetch(url, cache, refresh, delay):
    """The page at `url`, from the cache if it is there."""
    path = cache_path(url, cache)
    if not refresh and os.path.exists(path) and os.path.getsize(path) > 0:
        return open(path, encoding='utf-8', errors='replace').read(), True
    last = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': AGENT, 'Accept': 'text/html,application/json,*/*'})
            with urllib.request.urlopen(request, timeout=30) as response:
                page = response.read().decode(response.headers.get_content_charset() or 'utf-8', errors='replace')
            os.makedirs(cache, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(page)
            time.sleep(delay)
            return page, False
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            # Not there, or not ours to read: asking again will not change that.
            if isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 403, 404, 410):
                last = {401: 'it asks for a password', 404: 'it is not there', 410: 'it is not there'}.get(e.code, e)
                break
            # Asked to slow down, or a hiccup: wait longer each time.
            time.sleep(delay * 4 * (attempt + 1))
    raise RuntimeError(f'could not fetch {url}: {last}')


def text_of(fragment):
    """The text a reader sees in a piece of HTML, one line per block."""
    fragment = re.sub(r'(?is)<(script|style|noscript|svg)\b.*?</\1>', ' ', fragment)
    fragment = re.sub(r'(?i)<br\s*/?>|</(p|div|li|tr|td|th|h[1-6]|pre|section|article|span|a|button)>', '\n', fragment)
    fragment = re.sub(r'(?s)<[^>]+>', ' ', fragment)
    lines = [re.sub(r'[ \t\r\f\v ]+', ' ', html.unescape(line)).strip() for line in fragment.split('\n')]
    return [line for line in lines if line]


def paste_id(url):
    return url.rstrip('/').rsplit('/', 1)[1]


def paste(url, backend, cache, refresh, delay):
    """What the data service holds for the paste at `url`: its title and its Pokémon."""
    address = f'{backend.rstrip("/")}/api/paste/{paste_id(url)}?lang=english'
    for again in (refresh, True):
        text, cached = fetch(address, cache, again, delay)
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get('teams'), list):
            return data
        if not cached:
            break
        # What was kept from an earlier run is no good: ask once more.
    said = data.get('error') if isinstance(data, dict) else None
    raise RuntimeError(f'{address} did not answer with a team' + (f': {said}' if said else ''))


def export(mons):
    """A paste's Pokémon in Showdown's export format, one line per fact."""
    lines = []
    for mon in mons:
        if not isinstance(mon, dict) or not (mon.get('species') or mon.get('name')):
            continue
        head = str(mon.get('species') or mon['name']).strip()
        if str(mon.get('gender') or '').upper() in ('M', 'F'):
            head += f' ({str(mon["gender"]).upper()})'
        if mon.get('item'):
            head += ' @ ' + str(mon['item']).strip()
        lines.append(head)
        if mon.get('ability'):
            lines.append('Ability: ' + str(mon['ability']).strip())
        lines.append('Level: 50')
        # An open team sheet has no stat points. A paste someone made of their own team may:
        # they are kept if they are Champions stat points (66 in all, 32 at most in one stat).
        points = mon.get('evs') if isinstance(mon.get('evs'), dict) else {}
        points = {k.lower(): v for k, v in points.items() if isinstance(v, int) and not isinstance(v, bool)}
        given = [(points.get(key, 0), name) for key, name in STATS]
        if any(v > 0 for v, _ in given) and all(0 <= v <= 32 for v, _ in given) and sum(v for v, _ in given) <= 66:
            lines.append('EVs: ' + ' / '.join(f'{v} {name}' for v, name in given if v))
        if mon.get('nature'):
            lines.append(str(mon['nature']).strip() + ' Nature')
        lines += ['- ' + str(move).strip() for move in mon.get('moves') or [] if move]
        lines.append('')
    return lines[:-1]


def results(page):
    """The rows of the page's results tables that link to a team sheet, in order."""
    rows = []
    heading = ''
    # Headings and table rows, in the order they appear.
    for m in re.finditer(r'(?is)<h([1-6])\b[^>]*>(.*?)</h\1>|<tr\b[^>]*>(.*?)</tr>', page):
        if m.group(2) is not None:
            heading = ' '.join(text_of(m.group(2)))
            continue
        row = m.group(3)
        link = next((x for x in PASTE.finditer(row) if x.group(1) not in NOT_A_PASTE), None)
        if not link:
            continue
        cells = [' '.join(text_of(cell)) for cell in re.split(r'(?i)<t[dh]\b[^>]*>', row)[1:]]
        flat = ' '.join(cells)
        placing = next((int(c) for c in cells[:2] if re.fullmatch(r'\d{1,4}', c)), None)
        record = re.search(r'\b\d{1,2}-\d{1,2}(?:-\d{1,2})?\b', flat)
        country = re.search(r'flags/([A-Za-z]{2,3})-flag', row)
        species = re.findall(r'sprites/[^/"\']+/([A-Za-z0-9_-]+)\.(?:png|webp|gif)', row)
        rows.append({
            'section': heading,
            'placing': placing,
            'record': record.group(0) if record else '',
            'country': country.group(1).upper() if country else '',
            'species': species,
            'url': 'https://www.vrpastes.com/' + link.group(1),
            'cells': [c for c in cells if c],
        })
    if rows:
        return rows
    # No tables (the page is built differently from what this expects): take the links as they come.
    seen = []
    for link in PASTE.finditer(page):
        url = 'https://www.vrpastes.com/' + link.group(1)
        if link.group(1) not in NOT_A_PASTE and url not in seen:
            seen.append(url)
    return [{'section': '', 'placing': i + 1, 'record': '', 'country': '', 'species': [], 'url': url, 'cells': []}
            for i, url in enumerate(seen)]


def main():
    args = sys.argv[1:]
    if not args or args[0].startswith('-'):
        sys.exit(__doc__)
    url = args.pop(0)
    opts = {'--out': None, '--top': None, '--delay': '1.0', '--cache': None, '--backend': BACKEND}
    refresh = False
    while args:
        a = args.pop(0)
        if a == '--refresh':
            refresh = True
        elif a in opts and args:
            opts[a] = args.pop(0)
        else:
            sys.exit(f'unknown option {a}\n\n{__doc__}')
    slug = re.sub(r'[^a-z0-9]+', '-', url.rstrip('/').split('/')[-1].lower()).strip('-') or 'tournament'
    out = opts['--out'] or os.path.join('teams', 'raw', slug + '.json')
    cache = opts['--cache'] or os.path.join(os.path.dirname(out) or '.', 'cache')
    delay = float(opts['--delay'])

    page, _ = fetch(url, cache, refresh, delay)
    title = re.search(r'(?is)<title[^>]*>(.*?)</title>', page)
    event = html.unescape(re.sub(r'\s+', ' ', title.group(1))).strip() if title else slug
    rows = results(page)
    if opts['--top']:
        rows = rows[:int(opts['--top'])]
    if not rows:
        sys.exit(f'no links to vrpastes.com found on {url}; the page is saved in {cache} to look at')
    sections = []
    for row in rows:
        if row['section'] not in sections:
            sections.append(row['section'])
    print(f'{event}: {len(rows)} teams with a sheet, under {len(sections)} heading(s): '
          + '; '.join(f'"{s}" ({sum(r["section"] == s for r in rows)})' for s in sections))

    unread, whole = 0, 0
    for i, row in enumerate(rows):
        try:
            data = paste(row['url'], opts['--backend'], cache, refresh, delay)
        except RuntimeError as e:
            row['error'] = str(e)
            unread += 1
            print(f'  {i + 1}/{len(rows)} {row["url"]}: {e}')
            continue
        # "Eric Rios's 2027 Frankfurt Regional Championships OTS"
        row['title'] = str(data.get('title') or '').strip()
        owner = re.match(r"(.+?)['’]s? ", row['title'])
        row['player'] = owner.group(1) if owner else ''
        row['sheet'] = export(data['teams'])
        whole += sum(line.startswith('Ability: ') for line in row['sheet']) == 6
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f'  {i + 1}/{len(rows)} sheets read')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'source': url, 'event': event, 'teams': rows}, f, ensure_ascii=False, indent=1)
    print(f'wrote {out}: {whole} of {len(rows)} sheets list six Pokémon')
    if unread:
        print(f'{unread} of {len(rows)} sheets could not be fetched; run this again to try those again.')
    if not whole:
        sys.exit(f'No team came back from {opts["--backend"]}. The site may have moved its data: '
                 'see --backend at the top of this script.')
    print(f'next: cargo run --release --bin teampool -- {out} --play 1000')


if __name__ == '__main__':
    main()
