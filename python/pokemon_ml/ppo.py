"""PPO for self-play: collecting games, working out advantages, and the update.

Both sides of every game are played by the network being trained, so every
game gives two sets of decisions to learn from. The only reward is the
result: 1 to the winner and -1 to the loser when a game ends, nothing before,
nothing discounted.
"""
from dataclasses import dataclass

import torch

from .env import OBS_F, OBS_I, N_ACTIONS, part


@dataclass
class Config:
    steps: int = 64             # decisions every game advances between two updates
    epochs: int = 2             # passes over what was collected
    minibatch: int = 4096
    lr: float = 3e-4
    clip: float = 0.2           # how far one update may move the policy
    lam: float = 0.95           # generalized advantage estimation
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    # Jaxcalibur's "zero-avoiding" term: a pull towards the uniform policy that keeps every
    # legal action from dying out (the cross-entropy from uniform to the policy).
    uniform_coef: float = 0.001
    max_grad_norm: float = 0.5


class Rollout:
    """What `steps` decisions of `batch` observations need remembering, on `device`."""

    def __init__(self, steps, batch, device):
        z = lambda *shape, dtype=torch.float32: torch.zeros(steps, batch, *shape, dtype=dtype, device=device)
        self.f, self.i = z(OBS_F), z(OBS_I, dtype=torch.int16)
        self.mask0, self.mask1 = z(N_ACTIONS, dtype=torch.bool), z(N_ACTIONS, dtype=torch.bool)
        self.actions = z(2, dtype=torch.long)
        self.logp, self.value, self.reward = z(), z(), z()
        self.valid, self.done = z(dtype=torch.bool), z(dtype=torch.bool)
        self.last_value = torch.zeros(batch, device=device)
        self.steps, self.batch, self.device = steps, batch, device


def to_device(env, device):
    """The environment's current observations as tensors."""
    return (torch.from_numpy(env.f).to(device), torch.from_numpy(env.i).to(device), torch.from_numpy(env.mask).to(device))


@torch.no_grad()
def collect(env, model, roll):
    """Plays `roll.steps` decisions of every game with `model` on both sides."""
    model.eval()
    for t in range(roll.steps):
        f, i, mask = to_device(env, roll.device)
        actions, logp, value, mask0, mask1 = model.act(f, i, mask)
        roll.f[t], roll.i[t] = f, i
        roll.mask0[t], roll.mask1[t] = mask0.bool(), mask1.bool()
        roll.actions[t], roll.logp[t], roll.value[t] = actions, logp, value
        # A side with one thing it can do (waiting while the other replaces a Pokémon) has made no decision.
        roll.valid[t] = part(i, "i", "info")[:, 3] == 1
        env.step(actions.cpu().numpy())
        roll.reward[t] = torch.from_numpy(env.reward).to(roll.device).reshape(-1)
        roll.done[t] = torch.from_numpy(env.done).to(roll.device).bool().repeat_interleave(2)
    f, i, _ = to_device(env, roll.device)
    field = model.encode(f, i)[0]
    roll.last_value = model.value(field)


def advantages(value, reward, done, valid, last_value, lam):
    """Generalized advantage estimates and the returns to fit the value to, without
    discounting. All [steps, batch]; `done[t]` says the game ended on the action of step `t`
    (so what follows is another game), `valid[t]` that the side had a decision to make.
    A result that arrives on a step where a side had no decision goes to its last one."""
    adv = torch.zeros_like(value)
    next_value, next_adv, pending = last_value.clone(), torch.zeros_like(last_value), torch.zeros_like(last_value)
    for t in reversed(range(value.shape[0])):
        over = done[t]
        next_value = next_value.masked_fill(over, 0.0)
        next_adv = next_adv.masked_fill(over, 0.0)
        pending = pending.masked_fill(over, 0.0) + reward[t]
        a = pending + next_value - value[t] + lam * next_adv
        here = valid[t]
        adv[t] = torch.where(here, a, torch.zeros_like(a))
        next_value = torch.where(here, value[t], next_value)
        next_adv = torch.where(here, a, next_adv)
        pending = pending.masked_fill(here, 0.0)
    return adv, adv + value


def update(model, optimizer, roll, config):
    """One PPO update on a rollout. Returns what happened, for the log."""
    adv, returns = advantages(roll.value, roll.reward, roll.done, roll.valid, roll.last_value, config.lam)
    flat = lambda x: x.reshape(roll.steps * roll.batch, *x.shape[2:])
    f, i, mask0, mask1 = flat(roll.f), flat(roll.i), flat(roll.mask0), flat(roll.mask1)
    actions, old_logp, old_value, adv, returns = flat(roll.actions), flat(roll.logp), flat(roll.value), flat(adv), flat(returns)
    rows = flat(roll.valid).nonzero().squeeze(1)
    model.train()
    log = {k: 0.0 for k in ("policy_loss", "value_loss", "entropy", "kl", "clipped")}
    batches = 0
    for _ in range(config.epochs):
        order = rows[torch.randperm(len(rows), device=rows.device)]
        for start in range(0, len(order), config.minibatch):
            mb = order[start:start + config.minibatch]
            if len(mb) < 2:
                continue
            logp, entropy, spread, value = model.evaluate(f[mb], i[mb], mask0[mb], mask1[mb], actions[mb])
            a = adv[mb]
            a = (a - a.mean()) / (a.std() + 1e-8)
            ratio = (logp - old_logp[mb]).exp()
            policy_loss = -torch.min(ratio * a, ratio.clamp(1 - config.clip, 1 + config.clip) * a).mean()
            value_loss = 0.5 * (value - returns[mb]).pow(2).mean()
            loss = (policy_loss + config.value_coef * value_loss
                    - config.entropy_coef * entropy.mean() - config.uniform_coef * spread.mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                log["policy_loss"] += policy_loss.item()
                log["value_loss"] += value_loss.item()
                log["entropy"] += entropy.mean().item()
                log["kl"] += (old_logp[mb] - logp).mean().item()
                log["clipped"] += ((ratio - 1).abs() > config.clip).float().mean().item()
            batches += 1
    log = {k: v / max(batches, 1) for k, v in log.items()}
    # How much of the spread in results the value explained when the games were played.
    spread_of = returns[rows].var()
    log["explained"] = (1 - (returns[rows] - old_value[rows]).var() / spread_of).item() if spread_of > 0 else 0.0
    log["decisions"] = len(rows)
    return log


@torch.no_grad()
def play_baseline(env, model, kind, games, device, greedy=False):
    """The share of games `model` wins against a player that needs no model ("random" or
    "greedy"), over at least `games` games. It takes the first side of even-numbered games and
    the second of odd ones. Ties count as half."""
    model.eval()
    side = torch.arange(env.envs) % 2
    mine = (2 * torch.arange(env.envs) + side).numpy()
    wins = ties = played = 0
    while played < games:
        f, i, mask = to_device(env, device)
        actions = env.baseline(kind)
        actions[mine] = model.act(f[mine], i[mine], mask[mine], greedy=greedy)[0].cpu().numpy()
        env.step(actions)
        over = env.done.astype(bool)
        result = env.reward[over, side.numpy()[over]]
        wins, ties, played = wins + int((result > 0).sum()), ties + int((result == 0).sum()), played + int(over.sum())
    return (wins + 0.5 * ties) / played
