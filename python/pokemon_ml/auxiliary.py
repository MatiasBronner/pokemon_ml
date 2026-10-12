"""Training-only labels. Never concatenate these onto policy observations."""

import torch
import torch.nn.functional as F
from .env import part, PREVIEW
from .model import OPP_ACTIONS


def labels(i, actions):
    batch = len(i)
    opposite = torch.arange(batch, device=i.device) ^ 1
    private = part(i.long(), "i", "mon_ids")[opposite, :6]
    moves = part(i.long(), "i", "move_ids")[opposite]
    a = actions[opposite]
    rows = torch.arange(batch, device=i.device)
    result = []
    for pos in range(2):
        x = a[:, pos]
        mv = moves[rows, (4 * pos + x // 10).clamp(0, 7)]
        code = mv * 10 + x % 10
        code = torch.where(x < 40, code, OPP_ACTIONS - 7 + (x - 40).clamp(0, 6))
        result.append(code)
    target = torch.stack(result, -1)
    # Preview selections are a different action space; no fake move labels.
    valid = (part(i, "i", "info")[opposite, 0] != PREVIEW) & (
        part(i, "i", "info")[opposite, 3] == 1
    )
    return target, valid, private


def loss(outputs, obs_i, target, valid, private, obs_f=None):
    zero = (
        next(iter(outputs.values())).sum() * 0
        if outputs
        else obs_i.new_zeros((), dtype=torch.float32)
    )
    losses = {"opponent_loss": zero, "hidden_loss": zero}
    if "opponent" in outputs and valid.any():
        losses["opponent_loss"] = F.cross_entropy(
            outputs["opponent"][valid].flatten(0, 1), target[valid].flatten()
        )
    if "item" in outputs:
        public = part(obs_i.long(), "i", "mon_ids")[:, 6:]
        present = (public[..., 0] >= 2) & (public[..., 0] == private[..., 0])
        if obs_f is not None:
            present &= part(obs_f, "f", "mon_feats")[:, 6:, 27] == 0
        terms = []
        for key, col in [("item", 1), ("ability", 3)]:
            unknown = present & (public[..., col] == 1) & (private[..., col] >= 2)
            if unknown.any():
                terms.append(
                    F.cross_entropy(outputs[key][unknown], private[..., col][unknown])
                )
        vocab = outputs["moves"].shape[-1]
        known = F.one_hot(public[..., 4:8].clamp(0, vocab - 1), vocab).sum(-2).bool()
        truth = (
            F.one_hot(private[..., 4:8].clamp(0, vocab - 1), vocab)
            .sum(-2)
            .clamp(max=1)
            .float()
        )
        eligible = present[..., None] & ~known
        eligible[..., :2] = False
        # Illusion can make an opponent's apparent roster identity ambiguous.
        disguise = part(obs_i, "i", "mon_ids")[:, 6:, 0] != private[..., 0]
        eligible &= ~disguise[..., None]
        if eligible.any():
            raw = F.binary_cross_entropy_with_logits(
                outputs["moves"], truth, reduction="none"
            )
            positive = eligible & (truth == 1)
            negative = eligible & (truth == 0)
            balanced = [raw[m].mean() for m in (positive, negative) if m.any()]
            terms.append(torch.stack(balanced).mean())
        if terms:
            losses["hidden_loss"] = torch.stack(terms).mean()
    return losses


@torch.no_grad()
def metrics(outputs, obs_i, target, valid, private, obs_f):
    out = {}
    if "opponent" in outputs and valid.any():
        out["opponent_accuracy"] = float(
            (outputs["opponent"][valid].argmax(-1) == target[valid]).float().mean()
        )
    if "item" in outputs:
        public = part(obs_i.long(), "i", "mon_ids")[:, 6:]
        present = (public[..., 0] == private[..., 0]) & (public[..., 0] >= 2)
        present &= part(obs_f, "f", "mon_feats")[:, 6:, 27] == 0
        for name, col in [("item", 1), ("ability", 3)]:
            eligible = present & (public[..., col] == 1) & (private[..., col] >= 2)
            if eligible.any():
                out["hidden_" + name + "_accuracy"] = float(
                    (outputs[name][eligible].argmax(-1) == private[..., col][eligible])
                    .float()
                    .mean()
                )
    return out
