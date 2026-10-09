#!/usr/bin/env python3
"""Collects the teams of a tournament from Victory Road.

    scripts/scrape_teams.py https://victoryroad.pro/2027-frankfurt/ [--out teams/raw/2027-frankfurt.json]
                            [--top N] [--delay SECONDS] [--cache DIR] [--refresh]

Reads the tournament page, follows every team-sheet link in its results tables
(the "Export Team" links to vrpastes.com) and writes one JSON file with, for
each team, where it placed, who played it and the text of its sheet. It does
not interpret the sheets: `teampool` (cargo run --release --bin teampool)
turns the file into teams the engine can play, and says which it could not read.

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


def cache_path(url, cache):
    """Where the page at `url` is kept."""
    return os.path.join(cache, re.sub(r'[^A-Za-z0-9._-]+', '_', url.split('://', 1)[1]).strip('_') + '.html')


def fetch(url, cache, refresh, delay):
    """The page at `url`, from the cache if it is there."""
    path = cache_path(url, cache)
    if not refresh and os.path.exists(path) and os.path.getsize(path) > 0:
        return open(path, encoding='utf-8', errors='replace').read(), True
    last = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': AGENT, 'Accept': 'text/html,*/*'})
            with urllib.request.urlopen(request, timeout=30) as response:
                page = response.read().decode(response.headers.get_content_charset() or 'utf-8', errors='replace')
            os.makedirs(cache, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(page)
            time.sleep(delay)
            return page, False
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
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


def literals(script):
    """Every string literal in a piece of script or JSON, unescaped."""
    out = []
    for m in re.finditer(r'"((?:[^"\\\n]|\\.)*)"', script):
        try:
            out.append(json.loads('"' + m.group(1) + '"'))
        except ValueError:
            pass
    return out


def sheet_text(page):
    """Everything in a paste's page that could be part of the team, as short lines.

    The site is a script-driven one and may deliver the sheet as markup, as text
    inside its scripts, or as data inside its scripts. Rather than depend on which,
    this gathers the visible text and every string the scripts carry (two levels
    deep: the scripts quote their data). `teampool` knows the names of Pokémon,
    items, abilities and moves, and picks the team out of that.
    """
    lines = text_of(page)
    for script in re.findall(r'(?is)<script\b[^>]*>(.*?)</script>', page):
        for outer in literals(script):
            for inner in [outer] + literals(outer):
                lines.extend(part.strip() for part in inner.split('\n'))
    # A line of a team sheet is short. Keep the order, drop immediate repeats.
    kept = []
    for line in lines:
        line = re.sub(r'\s+', ' ', line).strip()
        if line and len(line) <= 90 and (not kept or kept[-1] != line):
            kept.append(line)
    return kept


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
    opts = {'--out': None, '--top': None, '--delay': '1.0', '--cache': None}
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

    unread = 0
    for i, row in enumerate(rows):
        try:
            paste, cached = fetch(row['url'], cache, refresh, delay)
        except RuntimeError as e:
            row['error'] = str(e)
            unread += 1
            print(f'  {i + 1}/{len(rows)} {row["url"]}: {e}')
            continue
        heading = re.search(r'(?is)<title[^>]*>(.*?)</title>', paste)
        row['title'] = html.unescape(re.sub(r'\s+', ' ', heading.group(1))).strip() if heading else ''
        # "Eric Rios's 2027 Frankfurt Regional Championships OTS – VR Pastes"
        owner = re.match(r"(.+?)['’]s? ", row['title'])
        row['player'] = owner.group(1) if owner else ''
        row['sheet'] = sheet_text(paste)
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f'  {i + 1}/{len(rows)} sheets read')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'source': url, 'event': event, 'teams': rows}, f, ensure_ascii=False, indent=1)
    print(f'wrote {out}')
    if unread:
        print(f'{unread} of {len(rows)} sheets could not be fetched; run this again to try those again.')
    print(f'next: cargo run --release --bin teampool -- {out} --play 1000')


if __name__ == '__main__':
    main()
