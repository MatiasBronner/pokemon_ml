"""Plays on a Pokémon Showdown server: against whoever the ladder finds, a
player who has been challenged, or one who challenges.

    # on a server of your own (see the README): two of these play each other
    python -m pokemon_ml.showdown --pool teams/2027-frankfurt.json --run runs/first \\
        --name botb --accept --games 20
    python -m pokemon_ml.showdown --pool teams/2027-frankfurt.json --run runs/first \\
        --name bota --challenge botb --games 20

    # on the public server, with an account of the bot's own
    python -m pokemon_ml.showdown --pool teams/2027-frankfurt.json --run runs/first \\
        --server sim3.psim.us --name MyBot --password ... --ladder --games 10

A battle arrives as text: a log of what happens and, at every decision, a
request listing the player's own Pokémon and what it may do. The Rust side
(`Follower`, `src/follow.rs`) turns that into the observation the network
was trained on and its reply back into Showdown's words, so the network
plays here exactly as it does in the simulator. This file is the rest: the
connection, logging in, finding battles, and keeping score.

The network is the one a run has saved (`--run`, with `--snapshot` for an
older version of it). Without `--run` the bot picks legal actions at random,
which is good for trying a connection out and for nothing else.
"""
import argparse
import asyncio
import json
import os
import random
import time
import urllib.parse
import urllib.request

import numpy as np

from . import _engine
from .env import N_ACTIONS, OBS_F, OBS_I, OBS_M

FORMAT = _engine.layout()["format"]
LOGIN = "https://play.pokemonshowdown.com/api/"


def userid(name):
    """Showdown's id for a name: its letters and digits, in lower case."""
    return "".join(c for c in name.lower() if c.isalnum())


class RandomPolicy:
    """Any legal pair of actions, each as likely as the next (the simulator's "random" player)."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, f, i, mask, preview):
        if preview:
            return self.rng.randrange(_engine.layout()["preview_actions"]), 0
        pairs = np.flatnonzero(mask[N_ACTIONS:])
        return divmod(int(self.rng.choice(pairs.tolist())), N_ACTIONS)


class ModelPolicy:
    """A trained network: `path` is a checkpoint or a snapshot of a run."""

    def __init__(self, path, device="cpu", greedy=False):
        import torch

        from . import model as model_lib

        saved = torch.load(path, map_location="cpu", weights_only=False)
        self.torch, self.device, self.greedy = torch, torch.device(device), greedy
        self.net = model_lib.Model(model_lib.Config(**saved["model_config"])).to(self.device).eval()
        try:
            self.net.load_state_dict(saved["model"])
        except RuntimeError as error:
            raise SystemExit(
                f"{path} was trained on an earlier layout of the observation and cannot play on this "
                f"version of the code; train again, or check out the version it was trained with.\n({error})")

    def act(self, f, i, mask, preview):
        t = lambda a: self.torch.from_numpy(a).unsqueeze(0).to(self.device)
        actions = self.net.act(t(f), t(i), t(mask), greedy=self.greedy)[0]
        return int(actions[0, 0]), int(actions[0, 1])


def saved_network(run, snapshot=None):
    """Where a run keeps the network to play: a snapshot by name, or else its latest checkpoint."""
    if snapshot:
        return os.path.join(run, "snapshots", snapshot + ".pt")
    return os.path.join(run, "checkpoint.pt")


class Battle:
    """One battle room: the follower of the bot's side of it, and what to send back."""

    def __init__(self, room, team, policy, open_sheets=True, sheets_wait=2.0, record=None):
        self.room, self.team, self.policy = room, team, policy
        self.open_sheets, self.sheets_wait = open_sheets, sheets_wait
        self.follower = None
        self.held = []  # lines that came before the bot knew which side it is
        self.pending = None  # a request not yet answered: (text, parsed)
        self.due = None  # when a Team Preview pick that is waiting for team sheets is made anyway
        self.asked_sheets = False
        self.sheets_shown = False
        self.over = False
        self.won = None
        self.opponent = ""
        self.names = {}
        self.decisions = 0
        self.refused = 0
        self.f = np.zeros(OBS_F, np.float32)
        self.i = np.zeros(OBS_I, np.int16)
        self.mask = np.zeros(OBS_M, np.uint8)
        self.record = open(record, "w", encoding="utf-8") if record else None

    def _follow(self, lines):
        if self.follower is None:
            self.held.extend(lines)
        elif lines:
            self.follower.lines("\n".join(lines))

    def receive(self, lines, me):
        """Takes in a message's lines; returns what to send to the room."""
        out, log = [], []
        for line in lines:
            if self.record:
                self.record.write(line + "\n")
            parts = line.split("|")
            kind = parts[1] if len(parts) > 1 else ""
            if kind == "request":
                text = line[len("|request|"):]
                if not text or text == "null":
                    continue
                self._follow(log)
                log = []
                parsed = json.loads(text)
                if self.follower is None:
                    self.follower = _engine.Follower(0 if parsed["side"]["id"] == "p1" else 1, json.dumps(self.team))
                    self._follow(self.held)
                    self.held = []
                self.pending = (text, parsed)
            elif kind == "error":
                # A move Showdown listed as usable and will not allow (a foe's Imprison comes to
                # light this way) is followed by the request again, put right. Anything else
                # refused is a fault here: let Showdown pick, so that the battle goes on.
                self.refused += 1
                if "[Unavailable choice]" not in line:
                    print(f"{self.room}: {line}", flush=True)
                    out.append("/choose default")
            else:
                if kind == "player" and len(parts) > 3 and parts[3]:
                    self.names[parts[2]] = parts[3]
                    if userid(parts[3]) != userid(me):
                        self.opponent = parts[3]
                if kind == "uhtml" and "otsrequest" in line and "acceptopenteamsheets" in line and not self.asked_sheets:
                    self.asked_sheets = True
                    out.append("/acceptopenteamsheets" if self.open_sheets else "/rejectopenteamsheets")
                if kind == "showteam":
                    self.sheets_shown = True
                if kind in ("win", "tie"):
                    self.over = True
                    self.won = None if kind == "tie" else userid(parts[2]) == userid(me)
                log.append(line)
        self._follow(log)
        return out + self.answer(time.monotonic())

    def answer(self, now):
        """The reply to the request in hand, if it is time to give one."""
        if self.pending is None or self.over or self.follower is None:
            return []
        text, parsed = self.pending
        if parsed.get("teamPreview") and self.open_sheets and self.asked_sheets and not self.sheets_shown:
            # With team sheets asked for, the other side's answer may be a moment coming: the
            # sheets, if they come, are worth having before the four are picked.
            if self.due is None:
                self.due = now + self.sheets_wait
            if now < self.due:
                return []
        self.pending = None
        self.follower.request(text)
        if parsed.get("wait"):
            return []
        legal = self.follower.observe(self.f, self.i, self.mask)
        preview = bool(parsed.get("teamPreview"))
        if legal > 1 or preview:
            first, second = self.policy.act(self.f, self.i, self.mask, preview)
            self.decisions += 1
        else:
            # One thing to do, and Showdown still wants to be told.
            first = int(np.flatnonzero(self.mask[:N_ACTIONS])[0])
            second = int(np.flatnonzero(self.mask[N_ACTIONS * (1 + first):N_ACTIONS * (2 + first)])[0])
        reply = self.follower.choice(first, second)
        rqid = parsed.get("rqid")
        return [f"/choose {reply}" + (f"|{rqid}" if rqid is not None else "")]

    def close(self):
        if self.record:
            self.record.close()


class Client:
    """A connection to a Showdown server, logged in under a name, playing battles."""

    def __init__(self, server, name, password=None, policy=None, teams=None, secure=None, open_sheets=True,
                 records=None, quiet=False, seed=None):
        if userid(name).startswith("guest") or not 0 < len(userid(name)) <= 18 or len(name) > 18:
            raise SystemExit(f"Showdown will not give the name {name!r}: a name is 1 to 18 characters and cannot begin with Guest")
        local = server.split(":")[0] in ("localhost", "127.0.0.1")
        self.secure = (not local) if secure is None else secure
        self.uri = f"{'wss' if self.secure else 'ws'}://{server}/showdown/websocket"
        self.name, self.password = name, password
        self.policy = policy or RandomPolicy(seed)
        self.teams = teams
        self.open_sheets = open_sheets
        self.records = records
        self.quiet = quiet
        self.rng = random.Random(seed)
        self.battles = {}
        self.results = []
        self.logged_in = False
        self.expecting = False  # a battle has been asked for and has not begun
        self.left = set()  # rooms of battles that are none of this session's
        # The public server takes a message every 0.6 seconds from an ordinary account.
        self.pace = 0.0 if local else 0.65
        self._last_sent = 0.0
        self.ws = None

    def say(self, *words):
        if not self.quiet:
            print(*words, flush=True)

    async def send(self, room, text):
        wait = self._last_sent + self.pace - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_sent = time.monotonic()
        await self.ws.send(f"{room}|{text}")

    def _assertion(self, challstr):
        """Showdown's login server vouches for a name: with the account's password, or without one for a name nobody has registered."""
        if self.password:
            data = urllib.parse.urlencode({"name": self.name, "pass": self.password, "challstr": challstr}).encode()
            with urllib.request.urlopen(LOGIN + "login", data, timeout=30) as reply:
                answer = json.loads(reply.read().decode()[1:])
            if not answer.get("actionsuccess"):
                raise SystemExit(f"Showdown did not accept the password for {self.name}")
            return answer["assertion"]
        query = urllib.parse.urlencode({"userid": userid(self.name), "challstr": challstr})
        with urllib.request.urlopen(LOGIN + "getassertion?" + query, timeout=30) as reply:
            answer = reply.read().decode()
        if answer.startswith(";"):
            raise SystemExit(f"the name {self.name} is registered: give its --password")
        return answer

    async def _login(self, challstr):
        # A server run with --no-security takes any name on its word.
        assertion = "" if not self.secure else await asyncio.to_thread(self._assertion, challstr)
        await self.send("", f"/trn {self.name},0,{assertion}")

    def _team(self):
        team = self.rng.choice(self.teams)
        problems = _engine.team_problems(json.dumps(team))
        if problems:
            raise SystemExit("a team of the pool is not legal: " + "; ".join(problems))
        return team

    async def _offer(self, mode, opponent):
        """Sets a team and looks for the next battle."""
        self.next_team = self._team()
        self.expecting = True
        await self.send("", "/utm " + _engine.packed_team(json.dumps(self.next_team)))
        if mode == "ladder":
            await self.send("", f"/search {FORMAT}")
        elif mode == "challenge":
            await self.send("", f"/challenge {opponent}, {FORMAT}")

    async def play(self, mode, games, opponent=None, timeout=None):
        """Plays `games` battles: `mode` is "ladder", "challenge" (of `opponent`) or "accept"
        (challenges, from `opponent` alone if one is named). Returns one result a battle."""
        began = time.monotonic()
        async with self._connect() as ws:
            self.ws = ws
            started = 0
            while len(self.results) < games:
                if timeout and time.monotonic() - began > timeout:
                    self.say(f"{self.name}: stopping after {timeout:.0f} seconds with {len(self.results)} battles played")
                    break
                if not self.logged_in and time.monotonic() - began > 30:
                    raise SystemExit(f"{self.name}: {self.uri} has not let it log in after 30 seconds")
                try:
                    message = await asyncio.wait_for(ws.recv(), 0.25)
                except asyncio.TimeoutError:
                    # Nothing new: a Team Preview pick that was waiting for sheets may be due.
                    for battle in list(self.battles.values()):
                        for reply in battle.answer(time.monotonic()):
                            await self.send(battle.room, reply)
                    continue
                lines = message.split("\n")
                room = lines[0][1:] if lines[0].startswith(">") else ""
                if room:
                    lines = lines[1:]
                if room.startswith("battle-"):
                    battle = self.battles.get(room)
                    if battle is None:
                        if room in self.left or not any(line.startswith("|init|battle") for line in lines):
                            continue
                        if not self.expecting:
                            # A battle left over from an earlier session under this name: it cannot
                            # be picked up half-way (which team was it, and what has been seen?).
                            self.say(f"{self.name}: forfeiting {room}, which this session did not start")
                            self.left.add(room)
                            await self.send(room, "/forfeit")
                            await self.send(room, "/leave")
                            continue
                        self.expecting = False
                        record = os.path.join(self.records, room + ".log") if self.records else None
                        battle = Battle(room, self.next_team, self.policy, self.open_sheets, record=record)
                        self.battles[room] = battle
                        started += 1
                    for reply in battle.receive(lines, self.name):
                        await self.send(room, reply)
                    if battle.over:
                        self.results.append({"room": room, "opponent": battle.opponent, "won": battle.won,
                                             "turns": battle.follower.turn if battle.follower else 0,
                                             "decisions": battle.decisions, "refused": battle.refused})
                        outcome = {True: "won", False: "lost", None: "tied"}[battle.won]
                        self.say(f"{self.name}: {outcome} against {battle.opponent} in {self.results[-1]['turns']} turns "
                                 f"({len(self.results)} of {games})")
                        battle.close()
                        del self.battles[room]
                        await self.send(room, "/leave")
                        if len(self.results) < games and mode != "accept":
                            await self._offer(mode, opponent)
                    continue
                for line in lines:
                    parts = line.split("|")
                    kind = parts[1] if len(parts) > 1 else ""
                    if kind == "challstr":
                        await self._login("|".join(parts[2:]))
                    elif kind == "updateuser" and len(parts) > 3:
                        named = parts[3] == "1" and userid(parts[2]) == userid(self.name)
                        if named and not self.logged_in:
                            self.logged_in = True
                            self.say(f"{self.name}: logged in to {self.uri}")
                            if mode == "accept":
                                self.next_team = self._team()
                                await self.send("", "/utm " + _engine.packed_team(json.dumps(self.next_team)))
                            else:
                                await self._offer(mode, opponent)
                    elif kind == "nametaken":
                        raise SystemExit(f"Showdown would not give the name {self.name}: {parts[-1]}")
                    elif kind == "pm" and len(parts) > 4 and parts[4].startswith("/challenge " + FORMAT):
                        sender = parts[2].strip()
                        mine = userid(parts[3]) == userid(self.name)
                        wanted = opponent is None or userid(sender) == userid(opponent)
                        if mode == "accept" and mine and wanted and started < games:
                            self.next_team = self._team()
                            self.expecting = True
                            await self.send("", "/utm " + _engine.packed_team(json.dumps(self.next_team)))
                            await self.send("", f"/accept {sender}")
                    elif kind == "popup":
                        self.say(f"{self.name}: Showdown says: {'|'.join(parts[2:])[:300]}")
        return self.results

    def _connect(self):
        import websockets

        return websockets.connect(self.uri, max_size=None, ping_interval=20)


def summary(results):
    """Wins, losses and ties of a list of results, and the share of battles won (a tie is half)."""
    won = sum(r["won"] is True for r in results)
    lost = sum(r["won"] is False for r in results)
    tied = len(results) - won - lost
    share = (won + 0.5 * tied) / max(len(results), 1)
    return won, lost, tied, share


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", required=True, help="the teams to play with (a pool, as `teampool` writes it)")
    p.add_argument("--team", type=int, help="play this team of the pool every game (default: one at random each game)")
    p.add_argument("--run", help="the training run whose network plays (default: random legal actions)")
    p.add_argument("--snapshot", help="a snapshot of the run by name, e.g. 000300 (default: its latest checkpoint)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--greedy", action="store_true", help="always the likeliest action, in place of drawing one")
    p.add_argument("--server", default="localhost:8000", help="host:port of the server (sim3.psim.us is the public one)")
    p.add_argument("--name", required=True, help="the account to play as")
    p.add_argument("--password", default=os.environ.get("SHOWDOWN_PASSWORD"),
                   help="its password, if it is registered (or set SHOWDOWN_PASSWORD)")
    how = p.add_mutually_exclusive_group(required=True)
    how.add_argument("--ladder", action="store_true", help="play whoever the ladder finds")
    how.add_argument("--challenge", metavar="USER", help="challenge this player")
    how.add_argument("--accept", action="store_true", help="wait for challenges and accept them")
    p.add_argument("--from", dest="only", metavar="USER", help="with --accept: only this player's challenges")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--closed-sheets", action="store_true", help="turn down open team sheets when they are offered")
    p.add_argument("--records", help="a directory to keep each battle's messages in, as they arrived")
    p.add_argument("--results", help="a file to add the results to, one JSON line a battle")
    p.add_argument("--seed", type=int)
    args = p.parse_args(argv)

    with open(args.pool, encoding="utf-8") as file:
        teams = [team["team"] for team in json.load(file)["teams"]]
    if args.team is not None:
        teams = [teams[args.team]]
    policy = None
    if args.run:
        policy = ModelPolicy(saved_network(args.run, args.snapshot), args.device, args.greedy)
    if args.records:
        os.makedirs(args.records, exist_ok=True)
    client = Client(args.server, args.name, args.password, policy, teams, open_sheets=not args.closed_sheets,
                    records=args.records, seed=args.seed)
    mode = "ladder" if args.ladder else "challenge" if args.challenge else "accept"
    results = asyncio.run(client.play(mode, args.games, args.challenge or args.only))
    if args.results:
        with open(args.results, "a", encoding="utf-8") as file:
            for r in results:
                file.write(json.dumps({"name": args.name, "run": args.run, "snapshot": args.snapshot, **r}) + "\n")
    won, lost, tied, share = summary(results)
    print(f"{args.name}: {won} won, {lost} lost, {tied} tied: {100 * share:.1f}% of {len(results)} battles")


if __name__ == "__main__":
    main()
