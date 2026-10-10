"""The network: one side's observation in, a policy and a value out.

Following Jaxcalibur's design, the observation becomes a short sequence of
tokens that a transformer mixes, and the policy is read off the tokens that
stand for the things it can pick:

    1   the field         weather, terrain, rooms, both sides' conditions, the turn
    12  Pokémon           the player's six and the opponent's six, by Team Preview order
    4   positions         the two Pokémon each side has on the field
    8   moves             the four moves of each of the player's two active Pokémon

A Pokémon token is a sum of embeddings: its species, item and ability, each of
its moves, and a projection of its numbers (HP, status, stats, what is known).
Species, moves and items are embedded as a learned vector per id plus a
projection of static dex data (types, base stats, a move's power, accuracy,
target and effects), so a rarely seen Pokémon or move starts from what it
is. There is no damage calculator and no usage statistics: only the state,
and what the order of moves has shown of each opposing Pokémon's Speed.
What the opponent has not shown is the id "unknown".

The policy:

    Team Preview   a score for each pair of the player's Pokémon as leads and as the pair
                   behind; one of the 90 ways to bring four is the sum of its two pairs
    a position     a logit for every (move, target, Mega or not), from the move's token and
                   the target's; a logit to switch to each of the player's Pokémon, from
                   that Pokémon's token; a logit to pass
    both positions the second is chosen given the first: an embedding of the first's action
                   is added to what the second's logits are computed from

The value is read off the field token: how the game will end, from -1 (lost) to 1 (won).
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import LAYOUT, N_ACTIONS, N_PREVIEW, PASS, PREVIEW, TABLES, part

ROSTER = LAYOUT["roster"]
NEG = -1e9


@dataclass
class Config:
    """Sizes of the network. The defaults are small on purpose: about a million parameters."""
    width: int = 128
    layers: int = 3
    heads: int = 4
    ff: int = 256
    dropout: float = 0.0


def mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


class Dex(nn.Module):
    """Ids to vectors: a learned embedding per id, plus static data about what the id is."""

    def __init__(self, count, width, table=None):
        super().__init__()
        self.learned = nn.Embedding(count, width, padding_idx=0)
        nn.init.normal_(self.learned.weight, std=0.02)
        with torch.no_grad():
            self.learned.weight[0].zero_()
        if table is not None:
            self.register_buffer("table", torch.as_tensor(table, dtype=torch.float32), persistent=False)
            self.static = nn.Linear(self.table.shape[1], width)
        else:
            self.table = None

    def vectors(self):
        """One vector per id. Id 0 ("nothing here") is the zero vector."""
        out = self.learned.weight
        if self.table is not None:
            out = out + self.static(self.table)
            out = torch.cat([torch.zeros_like(out[:1]), out[1:]])
        return out


class Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        d = self.config.width
        vocab = LAYOUT["vocab"]
        self.species = Dex(vocab["species"], d, TABLES["species"])
        self.items = Dex(vocab["items"], d, TABLES["items"])
        self.abilities = Dex(vocab["abilities"], d)
        self.moves = Dex(vocab["moves"], d, TABLES["moves"])
        size = lambda kind, name: LAYOUT[kind][name][1][-1]
        self.field_in = nn.Linear(size("f", "field"), d)
        self.mon_in = nn.Linear(size("f", "mon_feats"), d)
        self.act_in = nn.Linear(size("f", "act_feats"), d)
        self.move_in = nn.Linear(size("f", "move_feats"), d)
        self.lost_item = nn.Linear(d, d, bias=False)
        self.known_move = nn.Linear(d, d, bias=False)
        self.last_move = nn.Linear(d, d, bias=False)
        self.standing = nn.Linear(d, d, bias=False)
        self.user = nn.Linear(d, d, bias=False)
        self.mon_mix = nn.Sequential(nn.LayerNorm(d), mlp(d, 2 * d, d))
        # What a token is: the field, own Pokémon, their Pokémon, each of the four positions, a move of each position.
        self.role = nn.Parameter(torch.randn(9, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, self.config.heads, self.config.ff, self.config.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, self.config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

        self.value_head = mlp(d, d, 1)
        self.pair = mlp(d, d, 2)
        self.no_target = nn.Parameter(torch.randn(d) * 0.02)
        self.first_action = nn.Embedding(N_ACTIONS + 1, d)
        self.pick_move, self.pick_target, self.pick_context = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.move_out = nn.Sequential(nn.GELU(), nn.Linear(d, 2))
        self.switch_to, self.switch_context = nn.Linear(d, d), nn.Linear(d, d)
        self.switch_out = nn.Sequential(nn.GELU(), nn.Linear(d, 1))
        self.pass_out = mlp(d, d, 1)

        # Which pair of leads and which pair behind each Team Preview action is.
        preview = torch.as_tensor(TABLES["preview"], dtype=torch.long)
        pairs = [(a, b) for a in range(ROSTER) for b in range(a + 1, ROSTER)]
        index = {p: k for k, p in enumerate(pairs)}
        self.register_buffer("pair_a", torch.tensor([p[0] for p in pairs]), persistent=False)
        self.register_buffer("pair_b", torch.tensor([p[1] for p in pairs]), persistent=False)
        self.register_buffer("lead_pair", torch.tensor([index[(int(r[0]), int(r[1]))] for r in preview]), persistent=False)
        self.register_buffer("back_pair", torch.tensor([index[(int(r[2]), int(r[3]))] for r in preview]), persistent=False)

    # ------------------------------------------------------------------ tokens

    def encode(self, f, i):
        """Observations ([B, OBS_F] floats, [B, OBS_I] ids) to what the heads read: the field,
        the twelve Pokémon, the four positions and the eight moves, each [B, n, width]."""
        ids = i.long()
        species, items, abilities, moves = (m.vectors() for m in (self.species, self.items, self.abilities, self.moves))
        mon_ids = part(ids, "i", "mon_ids")
        mons = (species[mon_ids[..., 0]] + items[mon_ids[..., 1]] + self.lost_item(items[mon_ids[..., 2]])
                + abilities[mon_ids[..., 3]] + self.known_move(moves[mon_ids[..., 4:8]]).sum(-2)
                + self.mon_in(part(f, "f", "mon_feats")))
        mons = mons + self.mon_mix(mons)
        mons = mons + torch.cat([self.role[1].expand(ROSTER, -1), self.role[2].expand(ROSTER, -1)])

        # A position holds one of the Pokémon (or nobody: index 0 of a table with a blank row in front).
        act_ids = part(ids, "i", "act_ids")
        blank = torch.cat([torch.zeros_like(mons[:, :1]), mons], 1)
        who = act_ids[..., 0].unsqueeze(-1).expand(-1, -1, mons.shape[-1])
        acts = (self.standing(blank.gather(1, who)) + self.act_in(part(f, "f", "act_feats"))
                + self.last_move(moves[act_ids[..., 1]]) + self.role[3:7])

        move_ids = part(ids, "i", "move_ids")
        user = self.user(acts[:, :2]).repeat_interleave(4, 1)
        role = torch.cat([self.role[7].expand(4, -1), self.role[8].expand(4, -1)])
        move_tokens = moves[move_ids] + self.move_in(part(f, "f", "move_feats")) + user + role

        field = (self.field_in(part(f, "f", "field")) + self.role[0]).unsqueeze(1)
        h = self.norm(self.trunk(torch.cat([field, mons, acts, move_tokens], 1)))
        return h[:, 0], h[:, 1:13], h[:, 13:17], h[:, 17:25]

    # ------------------------------------------------------------------- heads

    def value(self, field):
        return self.value_head(field).squeeze(-1)

    def preview_logits(self, mons):
        own = mons[:, :ROSTER]
        scores = self.pair(own[:, self.pair_a] + own[:, self.pair_b])  # [B, 15, 2]
        return scores[:, self.lead_pair, 0] + scores[:, self.back_pair, 1]  # [B, 90]

    def position_logits(self, mons, acts, move_tokens, pos, first=None):
        """Logits over the N_ACTIONS actions of position `pos` (0 or 1). `first`: the action
        already chosen for the other position, for the second of the two."""
        context = acts[:, pos]
        if first is not None:
            context = context + self.first_action(first)
        mine = move_tokens[:, 4 * pos:4 * pos + 4]
        # Targets in the order of the action numbers: none, their first and second, own first and second.
        targets = torch.stack([self.no_target.expand_as(context), acts[:, 2], acts[:, 3], acts[:, 0], acts[:, 1]], 1)
        pick = (self.pick_move(mine)[:, :, None] + self.pick_target(targets)[:, None]
                + self.pick_context(context)[:, None, None])
        use = self.move_out(pick).flatten(1)  # [B, 4 moves × 5 targets × (plain, mega)]
        switch = self.switch_out(self.switch_to(mons[:, :ROSTER]) + self.switch_context(context)[:, None]).squeeze(-1)
        return torch.cat([use, switch, self.pass_out(context)], 1)

    # ------------------------------------------------------------------ acting

    @staticmethod
    def _masked(logits, mask):
        """Illegal actions out of the running. A row with nothing legal (it is not that row's
        kind of decision) is given one action so that it stays a distribution."""
        mask = mask.bool()
        empty = ~mask.any(-1, keepdim=True)
        fallback = torch.zeros_like(mask)
        fallback[:, PASS if mask.shape[1] == N_ACTIONS else 0] = True
        mask = mask | (empty & fallback)
        return logits.masked_fill(~mask, NEG), mask

    @staticmethod
    def _stats(logits, mask, action):
        """Log-probability of `action`, entropy, and the average log-probability of a legal action."""
        logp = F.log_softmax(logits, -1)
        chosen = logp.gather(1, action.unsqueeze(1)).squeeze(1)
        entropy = -(logp.exp() * logp).masked_fill(~mask, 0.0).sum(-1)
        spread = logp.masked_fill(~mask, 0.0).sum(-1) / mask.sum(-1)
        return chosen, entropy, spread

    @torch.no_grad()
    def act(self, f, i, mask, greedy=False):
        """Choose for every observation. Returns the actions [B, 2], their log-probability,
        the value, and the two masks the choice was made under (for `evaluate`)."""
        field, mons, acts, move_tokens = self.encode(f, i)
        preview = part(i, "i", "info")[:, 0] == PREVIEW
        mask0 = mask[:, :N_ACTIONS]
        mask1_all = mask[:, N_ACTIONS:].reshape(-1, N_ACTIONS, N_ACTIONS)
        choose = (lambda lg: lg.argmax(-1)) if greedy else (lambda lg: torch.multinomial(F.softmax(lg, -1), 1).squeeze(1))

        lg_p = self.preview_logits(mons)
        a_p = choose(lg_p)
        lg0, m0 = self._masked(self.position_logits(mons, acts, move_tokens, 0), mask0)
        a0 = choose(lg0)
        mask1 = mask1_all.gather(1, a0.view(-1, 1, 1).expand(-1, 1, N_ACTIONS)).squeeze(1)
        lg1, m1 = self._masked(self.position_logits(mons, acts, move_tokens, 1, a0), mask1)
        a1 = choose(lg1)

        logp_battle = self._stats(lg0, m0, a0)[0] + self._stats(lg1, m1, a1)[0]
        logp_preview = F.log_softmax(lg_p, -1).gather(1, a_p.unsqueeze(1)).squeeze(1)
        actions = torch.stack([torch.where(preview, a_p, a0), torch.where(preview, torch.zeros_like(a1), a1)], 1)
        return actions, torch.where(preview, logp_preview, logp_battle), self.value(field), mask0, mask1

    def evaluate(self, f, i, mask0, mask1, actions):
        """The log-probability the current network gives `actions`, the entropy of its choice,
        the average log-probability of a legal action (for the term that keeps every legal
        action alive), and the value."""
        field, mons, acts, move_tokens = self.encode(f, i)
        preview = part(i, "i", "info")[:, 0] == PREVIEW
        a0, a1 = actions[:, 0], actions[:, 1]

        lg_p = self.preview_logits(mons)
        everything = torch.ones_like(lg_p, dtype=torch.bool)
        p_logp, p_ent, p_spread = self._stats(lg_p, everything, a0.clamp(max=N_PREVIEW - 1))
        first = a0.clamp(max=N_ACTIONS - 1)
        lg0, m0 = self._masked(self.position_logits(mons, acts, move_tokens, 0), mask0)
        lg1, m1 = self._masked(self.position_logits(mons, acts, move_tokens, 1, first), mask1)
        logp0, ent0, spread0 = self._stats(lg0, m0, first)
        logp1, ent1, spread1 = self._stats(lg1, m1, a1.clamp(max=N_ACTIONS - 1))
        logp = torch.where(preview, p_logp, logp0 + logp1)
        entropy = torch.where(preview, p_ent, ent0 + ent1)
        spread = torch.where(preview, p_spread, spread0 + spread1)
        return logp, entropy, spread, self.value(field)

    def save_config(self):
        return asdict(self.config)
