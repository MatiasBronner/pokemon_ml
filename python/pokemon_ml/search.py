"""Search at play time: one turn ahead, as the simultaneous game it is.

The policy network on its own picks a pair of actions in one forward pass.
Here it only proposes. At a decision in battle:

1. The policy names the searching side's most likely pairs of actions
   (`candidates`), from its own observation.
2. `worlds` copies of the game are made in which what the searching side has
   not been shown is drawn anew (`Game::determinize`: which registered Pokémon
   wait unseen on the bench, and everyone's stat points). The searching side
   cannot tell any copy from the real game. In each, the other side's most
   likely `replies` come from the same network looking from the other side
   of that copy.
3. Every pair (ours, theirs) is played out one turn in the engine, `dice`
   times with dice of its own, and the value network scores where each ends
   up, from the searching side's view (a finished game scores its result).
   That is a payoff grid per world.
4. The grids are solved as one zero-sum game: one mixed strategy for the
   searching side, which does not know which world is the real one, against
   a best reply in each world, since the other side knows its own team. The
   pair played is drawn from that mixed strategy (or its most likely pair).

The search needs the opposing team's sheet, which open team sheets give: with
closed sheets their unrevealed moves, items and abilities would be the truth,
so those decisions are left to the policy. So are Team Preview and decisions
with only one legal pair. A model that reads decision history cannot be
searched yet: the positions a turn ahead have no history of their own.

    python -m pokemon_ml.search --checkpoint runs/x/checkpoint.pt --pool teams/p.json \\
        --opponent self --games 1000 --out runs/x/search-vs-self.json
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .env import Env, N_ACTIONS, OBS_F, OBS_I, OBS_M, PASS, PREVIEW, part
from .evaluation import load, outcome_summary


@dataclass
class Config:
    candidates: int = 8  # the searching side's pairs of actions considered
    replies: int = 8  # the other side's, in each world
    worlds: int = 4  # redrawn copies of what has not been shown
    dice: int = 4  # times each cell is played out (4 took no longer than 2 on a 3090)
    iterations: int = 1000  # of the equilibrium solver
    pick: str = "sample"  # "sample" from the mixed strategy, or play its "best" pair
    privileged: bool = False  # search the real game instead of redrawn ones (a cheat, for comparison)
    closed_sheets: bool = False  # search with closed sheets too (sees their real sets: a cheat)
    # Exploration, for games played to learn from (all off for evaluation and play). See `choose`.
    noise: float = 0.0  # share of Dirichlet noise in the odds that pick the extra pairs (with explore)
    noise_alpha: float = 0.0  # its concentration; 0: 10 / the number of legal pairs
    explore: int = 0  # candidates added beyond `candidates`, drawn uniformly from the other legal pairs
    explore_replies: int = 0  # the same for the other side's replies, in each world
    # piKL: how strongly each side's equilibrium strategy is held to its policy's own odds (see `solve`).
    # 0: not at all, the plain equilibrium. Larger trusts the policy more; 0.01 and up converge well.
    # 0.03 was best of 0.01-0.3 on Oscar: about 71% against the same network without search, over two
    # seeds, where the plain equilibrium (0) scored 64.5%.
    anchor: float = 0.03

    def __post_init__(self):
        if min(self.candidates, self.replies, self.worlds, self.dice, self.iterations) < 1:
            raise ValueError("candidates, replies, worlds, dice and iterations must be positive")
        if self.pick not in ("sample", "best"):
            raise ValueError('pick is "sample" or "best"')
        if not 0 <= self.noise <= 1 or self.noise_alpha < 0 or min(self.explore, self.explore_replies) < 0:
            raise ValueError("noise is a share in [0, 1]; noise_alpha, explore and explore_replies are not negative")
        if self.anchor < 0:
            raise ValueError("anchor is not negative")
        if self.noise > 0 and not (self.explore or self.explore_replies):
            raise ValueError("noise only chooses which extra pairs to explore: give explore or explore_replies too")


# ---------------------------------------------------------------- the solver


def solve(payoffs, iterations=1000, anchor=0.0, prior=None, priors=None):
    """Equilibrium of a zero-sum game in which the row player does not know
    which of several equally likely worlds it is in and the column player does.

    `payoffs`: one array per world, [rows, columns of that world], what the row
    player gets (the column player gets the negative). Returns the row
    player's mixed strategy, each world's column strategy, the value, and the
    gap: how much either player could still gain by deviating (0 at an
    equilibrium).

    With `anchor` 0: the plain equilibrium, by regret matching+ with linear
    averaging, which converges on zero-sum games.

    With `anchor` λ > 0 (piKL, as in Meta's Diplomacy work): each player
    maximises what it gets less λ times how far its strategy strays from a prior
    (KL divergence): the row player from `prior` (over its rows), the column
    player in each world from `priors[w]`. The payoffs a search fills in are
    noisy, and the more options it weighs the likelier one of them looks good by
    luck alone (the optimiser's curse). Held to the policy's odds, a pair the
    policy rates low gets weight only if it is found clearly better: at λ = 0.03,
    three points of win probability buy a factor of e. Solved by magnetic mirror
    descent with optimistic steps, whose last iterate converges to the unique
    anchored equilibrium (to a gap of 1e-9 or less for λ ≥ 0.01 in 1000
    iterations on grids of this size). The gap reported includes the penalty.
    """
    worlds = len(payoffs)
    rows = payoffs[0].shape[0]
    cols = max(a.shape[1] for a in payoffs)
    A = np.zeros((worlds, rows, cols))
    valid = np.zeros((worlds, cols), bool)
    for w, a in enumerate(payoffs):
        if a.shape[0] != rows:
            raise ValueError("every world must have the same rows")
        A[w, :, : a.shape[1]] = a
        valid[w, : a.shape[1]] = True
    if anchor > 0:
        return _anchored(A, valid, payoffs, iterations, anchor, prior, priors)
    regret_x, regret_y = np.zeros(rows), np.zeros((worlds, cols))
    sum_x, sum_y = np.zeros(rows), np.zeros((worlds, cols))
    uniform_y = valid / valid.sum(1, keepdims=True)

    def strategy(regret, fallback):
        total = regret.sum(-1, keepdims=True)
        return np.where(total > 0, regret / np.where(total > 0, total, 1), fallback)

    for t in range(1, iterations + 1):
        x = strategy(regret_x, np.full(rows, 1 / rows))
        y = strategy(regret_y, uniform_y)
        # Alternating updates: the rows answer the columns, then the columns the new rows.
        u_x = np.einsum("wkm,wm->k", A, y) / worlds
        regret_x = np.maximum(regret_x + u_x - x @ u_x, 0)
        x = strategy(regret_x, np.full(rows, 1 / rows))
        u_y = -np.einsum("k,wkm->wm", x, A)
        regret_y = np.maximum(regret_y + u_y - (y * u_y).sum(1, keepdims=True), 0) * valid
        sum_x += t * x
        sum_y += t * strategy(regret_y, uniform_y)
    x = sum_x / sum_x.sum()
    y = sum_y / sum_y.sum(1, keepdims=True)
    best_row = (np.einsum("wkm,wm->k", A, y) / worlds).max()
    against_x = np.einsum("k,wkm->wm", x, A)
    best_cols = np.where(valid, against_x, np.inf).min(1).mean()
    value = float(np.einsum("k,wkm,wm->", x, A, y) / worlds)
    return x, [y[w, : a.shape[1]] for w, a in enumerate(payoffs)], value, float(best_row - best_cols)


def _logsumexp(a, axis=-1):
    top = np.max(a, axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0)
    return (top + np.log(np.exp(a - top).sum(axis, keepdims=True))).squeeze(axis)


def _log_prior(p, valid):
    """log of a prior, renormalised over what is valid, with a floor so that nothing valid is impossible."""
    p = np.where(valid, np.asarray(p, float), 0)
    p = p / p.sum(-1, keepdims=True) if (p.sum(-1) > 0).all() else valid / valid.sum(-1, keepdims=True)
    p = np.where(valid, 0.999 * p + 0.001 / valid.sum(-1, keepdims=True), 0)
    with np.errstate(divide="ignore"):
        return np.where(valid, np.log(p), -np.inf)


def _anchored(A, valid, payoffs, iterations, lam, prior, priors):
    worlds, rows, cols = A.shape
    prior = np.full(rows, 1 / rows) if prior is None else prior
    padded = np.zeros((worlds, cols))
    for w in range(worlds):
        padded[w, : payoffs[w].shape[1]] = 1 if priors is None else priors[w]
    log_px = _log_prior(prior, np.ones(rows, bool))
    log_py = _log_prior(padded, valid)
    eta = 1.0 if lam < 0.1 else 0.5
    lx, ly = log_px.copy(), log_py.copy()
    last_ux, last_uy = np.zeros(rows), np.zeros((worlds, cols))
    for _ in range(iterations):
        x, y = np.exp(lx), np.exp(ly)
        u_x = np.einsum("wkm,wm->k", A, y) / worlds
        u_y = -np.einsum("k,wkm->wm", x, A)
        # Optimistic: step along twice this utility less the last, which damps the cycling of plain steps.
        step_x, step_y = 2 * u_x - last_ux, 2 * u_y - last_uy
        last_ux, last_uy = u_x, u_y
        lx = (lx + eta * lam * log_px + eta * step_x) / (1 + eta * lam)
        lx -= _logsumexp(lx)
        ly = np.where(valid, (ly + eta * lam * log_py + eta * step_y) / (1 + eta * lam), -np.inf)
        ly -= _logsumexp(ly)[:, None]
    x, y = np.exp(lx), np.exp(ly)

    def kl(p, log_q):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(p > 0, p * (np.log(np.where(p > 0, p, 1)) - np.where(p > 0, log_q, 0)), 0).sum(-1)

    u_x = np.einsum("wkm,wm->k", A, y) / worlds
    u_y = -np.einsum("k,wkm->wm", x, A)
    # Each side's best anchored answer to the other is worth λ·logsumexp(log prior + u/λ).
    gap_x = lam * _logsumexp(log_px + u_x / lam) - (x @ u_x - lam * kl(x, log_px))
    own_y = np.where(valid, y * u_y, 0).sum(1) - lam * kl(y, log_py)
    gap_y = lam * _logsumexp(np.where(valid, log_py + u_y / lam, -np.inf)) - own_y
    value = float(np.einsum("k,wkm,wm->", x, A, y) / worlds)
    return x, [y[w, : a.shape[1]] for w, a in enumerate(payoffs)], value, float(gap_x + gap_y.mean())


# ------------------------------------------------------------ the network


def joint_log_policy(model, f, i, mask):
    """log-probability of every pair of actions, [B, N_ACTIONS, N_ACTIONS], the
    way `Model.act` would choose them (illegal pairs -inf)."""
    _, mons, acts, move_tokens = model.encode(f, i)
    b = len(f)
    mask0 = mask[:, :N_ACTIONS].bool()
    mask1 = mask[:, N_ACTIONS:].reshape(b, N_ACTIONS, N_ACTIONS).bool()
    lg0, m0 = model._masked(model.position_logits(mons, acts, move_tokens, 0), mask0)
    logp0 = F.log_softmax(lg0, -1)
    first = torch.arange(N_ACTIONS, device=f.device).repeat(b)
    rep = lambda x: x.repeat_interleave(N_ACTIONS, 0)
    lg1 = model.position_logits(rep(mons), rep(acts), rep(move_tokens), 1, first)
    lg1, m1 = model._masked(lg1, mask1.reshape(b * N_ACTIONS, N_ACTIONS))
    logp1 = F.log_softmax(lg1, -1).reshape(b, N_ACTIONS, N_ACTIONS)
    m1 = m1.reshape(b, N_ACTIONS, N_ACTIONS)
    joint = logp0[:, :, None] + logp1
    return joint.masked_fill(~(m0[:, :, None] & m1), -np.inf)


def ranked(logp, legal, n):
    """The `n` most likely of the `legal` pairs under `logp` ([N_ACTIONS, N_ACTIONS]), most likely first."""
    legal = np.asarray(legal, np.int64).reshape(-1, 2)
    score = logp[legal[:, 0], legal[:, 1]]
    order = np.argsort(-score, kind="stable")[:n]
    return legal[order], score[order]


def choose(logp, legal, n, extra=0, noise=0.0, alpha=0.0, rng=None):
    """The pairs to search among the `legal` ones: the `n` the policy (`logp`) finds
    most likely, always, and `extra` more from the rest to explore.

    Without exploration a search can never try what the policy rates lower, and a
    network trained on the search would never learn it. The extras are drawn
    uniformly from the pairs left out or, with `noise`, are the rest's best under
    the policy's odds mixed with a share `noise` of Dirichlet noise (concentration
    `alpha`; 0 for 10 over the number of legal pairs), AlphaZero's noise at the root
    of its search.

    Exploration only adds. A first version let the noise re-rank which `n` were
    taken, which pushed the policy's own favourites out, for the other side too.
    On Oscar that cost 4.5 points (60.0% against 64.5%), while two uniform extras
    on top of the eight cost under one. AlphaZero's noise never takes a move away.

    Returns the pairs, the policy's log-probability of each, and which of them are
    there by exploration only.
    """
    legal = np.asarray(legal, np.int64).reshape(-1, 2)
    score = logp[legal[:, 0], legal[:, 1]]
    top = np.argsort(-score, kind="stable")[:n]
    order = top
    rest = np.setdiff1d(np.arange(len(legal)), top)
    if extra > 0 and len(rest):
        k = min(extra, len(rest))
        if noise > 0:
            finite = np.isfinite(score)
            odds = np.exp(score - score[finite].max()) if finite.any() else np.ones(len(legal))
            odds = odds / odds.sum()
            eta = rng.dirichlet(np.full(len(legal), alpha or 10 / len(legal)))
            mixed = ((1 - noise) * odds + noise * eta)[rest]
            picked = rest[np.argsort(-mixed, kind="stable")[:k]]
        else:
            picked = rng.choice(rest, k, replace=False)
        order = np.concatenate([top, picked])
    return legal[order], score[order], ~np.isin(order, top)


def values(model, f, i, outcome, device, batch=4096):
    """The value network's verdict on positions, [-1, 1], from the side they are observed from;
    a finished game's `outcome` instead."""
    out = []
    for k in range(0, len(f), batch):
        fb = torch.as_tensor(f[k : k + batch], device=device)
        ib = torch.as_tensor(i[k : k + batch], device=device)
        field = model.encode(fb, ib)[0]
        out.append(model.value(field).clamp(-1, 1).float().cpu().numpy())
    v = np.concatenate(out) if out else np.zeros(0, np.float32)
    return np.where(np.isnan(outcome), v, outcome)


# ------------------------------------------------------------ the search


@torch.no_grad()
def search(model, env, game, view, f, i, mask, config, seed, rng, device):
    """The pair of actions the searching side `view` plays in game `game` of `env`,
    given its observation (`f`, `i`, `mask`: one row each), and what the search found."""
    worlds = env.raw.worlds(game, view, config.worlds, seed, redraw=not config.privileged)
    own_legal = [set(map(tuple, worlds.legal(w, view))) for w in range(worlds.count)]
    legal = sorted(set.intersection(*own_legal))
    t = lambda x: torch.as_tensor(x, device=device)
    mine = joint_log_policy(model, t(f[None]), t(i[None]), t(mask[None]))[0].cpu().numpy()
    if len(legal) == 1:
        return np.asarray(legal[0]), dict(searched=False)
    own, own_logp, explored = choose(
        mine, legal, config.candidates, config.explore, config.noise, config.noise_alpha, rng
    )
    policy_top = ranked(mine, legal, 1)[0][0]

    fo = np.zeros((worlds.count, OBS_F), np.float32)
    io = np.zeros((worlds.count, OBS_I), np.int16)
    mo = np.zeros((worlds.count, OBS_M), np.uint8)
    worlds.observe(1 - view, fo, io, mo)
    theirs_logp = joint_log_policy(model, t(fo), t(io), t(mo)).cpu().numpy()
    chosen = [
        choose(
            theirs_logp[w], worlds.legal(w, 1 - view), config.replies,
            config.explore_replies, config.noise, config.noise_alpha, rng,
        )
        for w in range(worlds.count)
    ]
    replies = [c[0] for c in chosen]

    sizes = [len(own) * len(r) * config.dice for r in replies]
    total = sum(sizes)
    cf = np.zeros((total, OBS_F), np.float32)
    ci = np.zeros((total, OBS_I), np.int16)
    cm = np.zeros((total, OBS_M), np.uint8)
    outcome = np.zeros(total, np.float32)
    at = 0
    own32 = np.ascontiguousarray(own, np.int32)
    for w, (r, n) in enumerate(zip(replies, sizes)):
        worlds.expand(
            w, own32, np.ascontiguousarray(r, np.int32), config.dice, seed * 1000 + w,
            cf[at : at + n], ci[at : at + n], cm[at : at + n], outcome[at : at + n],
        )
        at += n
    v = values(model, cf, ci, outcome, device)
    payoffs, at = [], 0
    for r, n in zip(replies, sizes):
        payoffs.append(v[at : at + n].reshape(len(own), len(r), config.dice).mean(-1))
        at += n
    odds = lambda logp: np.exp(logp - logp[np.isfinite(logp)].max()) if np.isfinite(logp).any() else np.ones(len(logp))
    x, _, value, gap = solve(
        payoffs, config.iterations, config.anchor,
        prior=odds(own_logp), priors=[odds(c[1]) for c in chosen],
    )
    k = int(np.argmax(x)) if config.pick == "best" else int(rng.choice(len(x), p=x))
    prior = np.exp(own_logp - own_logp.max())
    return own[k], dict(
        searched=True,
        value=value,
        gap=gap,
        candidates=len(own),
        policy_top=bool((own[k] == policy_top).all()),
        mixed=float(1 - x.max()),
        # What exploration brought in: the equilibrium's weight on such pairs, and whether one was played.
        explored_weight=float(x[explored].sum()),
        played_explored=bool(explored[k]),
        # How far the equilibrium moved off the policy's own (renormalised) odds over the candidates.
        shift=float(0.5 * np.abs(x - prior / prior.sum()).sum()),
        cells=total,
    )


# ------------------------------------------------------------ evaluation


@torch.no_grad()
def duel(env, model, opponent, games, config, device, seed=0):
    """`model` with search against `opponent` (a model, played straight from its policy, or a
    scripted player's name), each game's first episode only, sides alternating."""
    if model.config.history or (not isinstance(opponent, str) and opponent.config.history):
        raise ValueError("search does not support models that read decision history")
    model.eval()
    if not isinstance(opponent, str):
        opponent.eval()
    rng = np.random.default_rng(seed)
    side = np.arange(env.envs) % 2
    mine = 2 * np.arange(env.envs) + side
    theirs = 2 * np.arange(env.envs) + 1 - side
    results, stats = [], []
    active = np.arange(env.envs) < min(env.envs, games)
    batch, decisions, start = 0, 0, time.time()
    initial_seed = env.seed
    while len(results) < games:
        f, i, mask = (torch.as_tensor(x, device=device) for x in (env.f, env.i, env.mask))
        if isinstance(opponent, str):
            actions = env.baseline(opponent).astype(np.int64)
        else:
            actions = np.zeros((2 * env.envs, 2), np.int64)
            actions[theirs] = opponent.act(f[theirs], i[theirs], mask[theirs])[0].cpu().numpy()
        actions[mine] = model.act(f[mine], i[mine], mask[mine])[0].cpu().numpy()
        info = part(env.i, "i", "info")
        sheets = env.raw.open_sheets()
        for g in np.flatnonzero(active):
            row = mine[g]
            if info[row, 0] == PREVIEW or info[row, 3] != 1:
                continue
            if not (sheets[g] or config.closed_sheets):
                continue
            decisions += 1
            pair, found = search(
                model, env, int(g), int(side[g]), env.f[row], env.i[row], env.mask[row],
                config, seed * 1_000_003 + decisions, rng, device,
            )
            actions[row] = pair
            if found.get("searched"):
                stats.append(found)
        env.step(actions)
        for g in np.flatnonzero(env.done.astype(bool) & active):
            active[g] = False
            if len(results) < games:
                results.append(int(env.reward[g, side[g]]))
        if not active.any() and len(results) < games:
            batch += 1
            env = env.restart(initial_seed + batch)
            active[:] = np.arange(env.envs) < min(env.envs, games - len(results))
    mean = lambda key: float(np.mean([s[key] for s in stats])) if stats else None
    return {
        **outcome_summary(results),
        "searched_decisions": len(stats),
        "seconds": time.time() - start,
        "seconds_per_search": (time.time() - start) / max(len(stats), 1),
        "mean_equilibrium_gap": mean("gap"),
        "played_policy_top_pair": mean("policy_top"),
        "mean_mixed": mean("mixed"),
        "mean_shift_from_policy": mean("shift"),
        "mean_cells": mean("cells"),
        "mean_explored_weight": mean("explored_weight"),
        "played_explored_pair": mean("played_explored"),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="the network that searches")
    p.add_argument(
        "--opponent",
        default="self",
        help='"self" (the same network without search), random, greedy, lookahead or a checkpoint path',
    )
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=9000001)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--open-sheets", type=float, default=1.0, help="games with closed sheets are played by the policy")
    p.add_argument("--max-turns", type=int, default=100)
    defaults = Config()
    for name in ("candidates", "replies", "worlds", "dice", "iterations"):
        p.add_argument(f"--{name}", type=int, default=getattr(defaults, name))
    p.add_argument("--pick", choices=("sample", "best"), default=defaults.pick)
    p.add_argument("--privileged", action="store_true", help="search the real game (sees hidden information)")
    p.add_argument("--closed-sheets", action="store_true", help="search closed-sheet games too (sees their sets)")
    p.add_argument("--noise", type=float, default=defaults.noise, help="Dirichlet noise in the odds that choose the extra pairs (needs --explore); AlphaZero: 0.25")
    p.add_argument("--noise-alpha", type=float, default=defaults.noise_alpha, help="Dirichlet concentration; 0: 10 / legal pairs")
    p.add_argument("--explore", type=int, default=defaults.explore, help="extra candidate pairs drawn at random")
    p.add_argument("--explore-replies", type=int, default=defaults.explore_replies, help="extra replies drawn at random")
    p.add_argument("--anchor", type=float, default=defaults.anchor, help="piKL: hold the equilibrium to the policy's odds (0: off)")
    a = p.parse_args(argv)
    from .train import pick_device
    from .checkpoint import file_hash

    config = Config(
        candidates=a.candidates, replies=a.replies, worlds=a.worlds, dice=a.dice, iterations=a.iterations,
        pick=a.pick, privileged=a.privileged, closed_sheets=a.closed_sheets,
        noise=a.noise, noise_alpha=a.noise_alpha, explore=a.explore, explore_replies=a.explore_replies,
        anchor=a.anchor,
    )
    device = pick_device(a.device)
    model = load(a.checkpoint, device)
    if a.opponent == "self":
        opponent = load(a.checkpoint, device)
    elif a.opponent in ("random", "greedy", "lookahead"):
        opponent = a.opponent
    else:
        opponent = load(a.opponent, device)
    env = Env(a.pool, a.envs, a.seed, a.open_sheets, False, max_turns=a.max_turns, threads=a.threads)
    report = duel(env, model, opponent, a.games, config, device, seed=a.seed)
    report.update(
        search=asdict(config),
        seed=a.seed,
        open_sheets=a.open_sheets,
        opponent=a.opponent,
        pool_sha256=file_hash(a.pool),
        checkpoint_sha256=file_hash(a.checkpoint),
        opponent_sha256=None if a.opponent in ("self", "random", "greedy", "lookahead") else file_hash(a.opponent),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("games", "score", "score_interval_95", "searched_decisions", "seconds_per_search")}))


if __name__ == "__main__":
    main()
