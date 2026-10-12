"""Model-side feature ablations and bounded public observation history.

History records decision snapshots, not simulator-private actions or an exact
move event log. The same builder is used for training, evaluation and Showdown.
"""

import torch
from .env import LAYOUT, part

HISTORY_F = 3 + 4 * 16  # phase, sheet mode, turn + presence/HP/status/boosts per active
FEATURE_SCHEMA = {
    "version": 1,
    "mon_speed": [46, 52],
    "active_order": [119, 125],
    "history": "public decision snapshots: phase,sheets,turn,active presence/HP/status/boosts/species/last-move",
}
assert LAYOUT["f"]["mon_feats"][1][-1] == 52 and LAYOUT["f"]["act_feats"][1][-1] == 125

HISTORY_I = 8  # species and last visibly attempted move, per active


def policy_features(f, use_speed=True):
    if use_speed:
        return f
    f = f.clone()
    # Remove resolver-derived range/item possibilities, preserving known stats.
    part(f, "f", "mon_feats")[..., -6:] = 0
    # The last six active features encode two cross-field ordering comparisons.
    part(f, "f", "act_feats")[..., -6:] = 0
    return f


class History:
    def __init__(self, batch, length, device):
        self.length = length
        self.f = torch.zeros(batch, length, HISTORY_F, device=device)
        self.i = torch.zeros(batch, length, HISTORY_I, dtype=torch.long, device=device)
        self.valid = torch.zeros(batch, length, dtype=torch.bool, device=device)

    def tensors(self, rows=None):
        return tuple(
            x if rows is None else x[rows] for x in (self.f, self.i, self.valid)
        )

    def append(self, f, i, active=None):
        if not self.length:
            return
        if active is None:
            active = torch.ones(len(f), dtype=torch.bool, device=f.device)
        field = part(f, "f", "field")
        acts = part(f, "f", "act_feats")
        ids = part(i.long(), "i", "act_ids")
        mons = part(i.long(), "i", "mon_ids")
        species = torch.cat([torch.zeros_like(mons[:, :1, 0]), mons[..., 0]], 1)
        hf = torch.cat(
            [
                field[:, :3].argmax(-1, keepdim=True).float() / 2,
                field[:, 3:5],
                acts[..., :16].flatten(1),
            ],
            1,
        )
        hi = torch.stack([species.gather(1, ids[..., 0]), ids[..., 1]], -1).flatten(1)
        for x, new in ((self.f, hf), (self.i, hi), (self.valid, active)):
            shifted = torch.cat([x[:, 1:], new[:, None]], 1)
            x[active] = shifted[active]

    def reset(self, done):
        self.f[done], self.i[done], self.valid[done] = 0, 0, False
