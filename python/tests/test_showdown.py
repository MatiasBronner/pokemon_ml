"""The Showdown client.

Most of it is tried without a server: battles that were played in Pokémon
Showdown (the Rust tests' fixture) are fed to it message by message, as a
connection would deliver them. The last test plays real battles through a
server and runs only where one is named:

    cd oracle/pokemon-showdown && node pokemon-showdown start --no-security 8000
    SHOWDOWN_SERVER=localhost:8000 pytest python/tests/test_showdown.py
"""
import asyncio
import json
import os
import random

import numpy as np
import pytest

from pokemon_ml import _engine, showdown
from pokemon_ml.env import N_ACTIONS, OBS_F, OBS_I, OBS_M

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(REPO, "tests", "fixtures", "followed_battles.jsonl")


def recorded():
    """The recorded battles whose teams a ladder would allow (the recorder's can have a species
    twice), and that are there to the end (one of the fixture's is cut short)."""
    with open(FIXTURE, encoding="utf-8") as file:
        cases = [json.loads(line) for line in file]
    whole = lambda c: any(line.startswith("|win|") for line in c["steps"][-1]["after"]["log"])
    return [c for c in cases if whole(c) and all(len({m["species"] for m in team}) == len(team) for team in c["rosters"])]


def own_lines(side, log):
    """What one player is sent of a battle's whole log: of each pair with a private and a public version, its own."""
    out, k = [], 0
    while k < len(log):
        if log[k].startswith("|split|"):
            out.append(log[k + 1] if log[k] == f"|split|p{side + 1}" else log[k + 2])
            k += 3
        else:
            out.append(log[k])
            k += 1
    return out


def well_formed(reply):
    assert reply.startswith("/choose "), reply
    parts = reply[len("/choose "):].split("|")[0].split(", ")
    assert len(parts) == 2 and all(p.split(" ")[0] in ("move", "switch", "pass") for p in parts), reply


def test_a_recorded_battle_goes_through_the_client():
    cases = recorded()
    assert len(cases) >= 10
    decisions = 0
    for case in cases:
        for side in range(2):
            me = f"P{side + 1}"
            battle = showdown.Battle("battle-test-1", case["rosters"][side], showdown.RandomPolicy(seed=case["id"]))
            # The log so far arrives first, then the request: the bot learns which side it is from that.
            # (All it has to say before that is whether it will show its team sheet.)
            early = battle.receive(own_lines(side, case["initial"]["log"]), me)
            assert early in ([], ["/acceptopenteamsheets"])
            replies = battle.receive(["|request|" + json.dumps(case["initial"]["requests"][side])], me)
            assert len(replies) == 1 and battle.follower.side == side
            well_formed(replies[0])
            for step in case["steps"]:
                # (The battle went the way it was recorded, whatever this bot would have done.)
                told = step["after"]
                lines = own_lines(side, told["log"])
                request = told["requests"][side] if told["requests"] else None
                if request:
                    lines.append("|request|" + json.dumps(request))
                replies = battle.receive(lines, me)
                if request and not request.get("wait"):
                    assert len(replies) == 1, (case["id"], replies)
                    well_formed(replies[0])
                else:
                    assert replies == []
            last = [line for line in case["steps"][-1]["after"]["log"] if line.startswith("|win|")]
            assert battle.over and battle.won == (last[0] == f"|win|{me}")
            assert battle.follower.ended and battle.follower.winner == (0 if last[0] == "|win|P1" else 1)
            assert battle.opponent == f"P{2 - side}"
            decisions += battle.decisions
    assert decisions > 200


def test_a_choice_showdown_takes_back_is_made_again():
    """A switch that turns out to be barred: Showdown says so and sends the request again, put right."""
    for case in recorded():
        for side in range(2):
            for k, step in enumerate(case["steps"]):
                request = step["after"]["requests"][side] if step["after"]["requests"] else None
                unsure = [p for p, a in enumerate((request or {}).get("active", [])) if a.get("maybeTrapped")]
                bench = [m for m in (request or {}).get("side", {}).get("pokemon", [])[2:] if not m["condition"].endswith("fnt")]
                if not unsure or not bench:
                    continue
                me = f"P{side + 1}"
                battle = showdown.Battle("battle-test-2", case["rosters"][side], showdown.RandomPolicy(seed=1))
                battle.receive(own_lines(side, case["initial"]["log"]), me)
                battle.receive(["|request|" + json.dumps(case["initial"]["requests"][side])], me)
                for earlier in case["steps"][:k]:
                    battle.receive(own_lines(side, earlier["after"]["log"])
                                   + ["|request|" + json.dumps(earlier["after"]["requests"][side])], me)
                first = battle.receive(own_lines(side, step["after"]["log"]) + ["|request|" + json.dumps(request)], me)
                well_formed(first[0])
                # Refused: nothing is sent until the request comes again.
                assert battle.receive(["|error|[Unavailable choice] Can't switch: The active Pokémon is trapped"], me) == []
                fixed = json.loads(json.dumps(request))
                del fixed["active"][unsure[0]]["maybeTrapped"]
                fixed["active"][unsure[0]]["trapped"] = True
                again = battle.receive(["|request|" + json.dumps(fixed)], me)
                well_formed(again[0])
                assert not again[0][len("/choose "):].split(", ")[unsure[0]].startswith("switch")
                assert battle.refused == 1
                return
    pytest.fail("no battle of the fixture has a Pokémon that may be trapped")


def test_the_follower_fills_the_arrays_training_fills():
    case = recorded()[0]
    follower = _engine.Follower(0, json.dumps(case["rosters"][0]))
    f, i, mask = np.zeros(OBS_F, np.float32), np.zeros(OBS_I, np.int16), np.zeros(OBS_M, np.uint8)
    # Team Preview: the six on the other side are shown, and every pick of four is open.
    log = own_lines(0, case["initial"]["log"])
    cut = next(k for k, line in enumerate(log) if line.startswith("|teamsize|"))
    follower.lines("\n".join(log[:cut]))
    assert follower.observe(f, i, mask) == _engine.layout()["preview_actions"]
    assert follower.choice(0, 0) == "team 1234"
    assert not follower.ended
    # The first turn: a choice for each of two Pokémon, and the arrays say what of.
    follower.lines("\n".join(log[cut:]))
    follower.request(json.dumps(case["initial"]["requests"][0]))
    legal = follower.observe(f, i, mask)
    assert follower.phase == "move" and follower.turn == 1 and legal > 1
    assert np.abs(f).sum() > 0 and (i != 0).sum() > 10
    first = int(np.flatnonzero(mask[:N_ACTIONS])[0])
    second = int(np.flatnonzero(mask[N_ACTIONS * (1 + first):N_ACTIONS * (2 + first)])[0])
    well_formed("/choose " + follower.choice(first, second))


def test_a_team_is_packed_as_showdown_reads_it(pool):
    with open(pool, encoding="utf-8") as file:
        team = json.load(file)["teams"][0]["team"]
    packed = _engine.packed_team(json.dumps(team))
    mons = packed.split("]")
    assert len(mons) == len(team)
    for mon in mons:
        fields = mon.split("|")
        # name, species, item, ability, moves, nature, EVs, gender, IVs, shiny, level, the rest
        assert len(fields) == 12 and fields[10] == "50"
        assert 1 <= len(fields[4].split(",")) <= 4 and len(fields[6].split(",")) == 6


@pytest.mark.skipif(not os.environ.get("SHOWDOWN_SERVER"), reason="set SHOWDOWN_SERVER to a server to play on")
def test_two_bots_play_each_other_on_a_server(pool):
    with open(pool, encoding="utf-8") as file:
        teams = [t["team"] for t in json.load(file)["teams"]]
    server, tag = os.environ["SHOWDOWN_SERVER"], random.randrange(10**6)
    host = showdown.Client(server, f"homebot{tag}", teams=teams, quiet=True, seed=1)
    guest = showdown.Client(server, f"awaybot{tag}", teams=teams, quiet=True, seed=2)

    async def both():
        waiting = asyncio.create_task(host.play("accept", 3, timeout=300))
        await asyncio.sleep(1.0)
        return await asyncio.gather(waiting, guest.play("challenge", 3, host.name, timeout=300))

    hosted, visited = asyncio.run(both())
    assert len(hosted) == len(visited) == 3
    for a, b in zip(hosted, visited):
        # One battle, seen from both ends: the same room, and one winner.
        assert a["room"] == b["room"] and a["turns"] == b["turns"] and a["turns"] > 0
        assert (a["won"], b["won"]) in ((True, False), (False, True), (None, None))
        assert a["refused"] == 0 and b["refused"] == 0
