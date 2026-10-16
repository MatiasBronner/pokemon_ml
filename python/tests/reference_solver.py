"""The equilibrium solver as it was before batching (merged in #53), kept verbatim so the
batched solver can be checked against it. Not used by anything but the tests."""
import numpy as np


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


