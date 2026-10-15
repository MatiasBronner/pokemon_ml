"""Read-only decision diagnostics from the actor's public observation only."""

import numpy as np
import torch

from .env import N_ACTIONS, NAMES, PREVIEW, TABLES, part


def describe_action(ids, action, pos=0, preview=False):
    mons = part(ids, "i", "mon_ids")
    species = lambda index: NAMES["species"][int(mons[index, 0])]
    if preview:
        picked = [int(x) for x in TABLES["preview"][action]]
        return dict(
            kind="preview",
            roster_indices=picked,
            leads=[species(x) for x in picked[:2]],
            reserves=[species(x) for x in picked[2:]],
        )
    if action < 40:
        move = int(part(ids, "i", "move_ids")[4 * pos + action // 10])
        target = (action // 2) % 5
        return dict(
            kind="move",
            move=NAMES["moves"][move],
            move_slot=action // 10,
            target=[
                "automatic",
                "opponent-left",
                "opponent-right",
                "own-left",
                "own-right",
            ][target],
            mega=bool(action % 2),
        )
    if action < 46:
        return dict(
            kind="switch", roster_index=action - 40, species=species(action - 40)
        )
    return dict(kind="pass")


@torch.no_grad()
def decision_details(model, f, i, mask, actions, history=None):
    """One observation. Slot-1 alternatives are conditional on chosen slot 0."""
    _, mons, acts, moves = model.encode(f, i, history)
    preview = int(part(i, "i", "info")[0, 0]) == PREVIEW
    ids = i[0].cpu().numpy()

    def summarize(logits, legal, chosen, pos, is_preview=False):
        probabilities = logits.softmax(-1)[0]
        count = int(legal[0].sum())
        values, indices = probabilities.topk(min(5, count))
        return dict(
            chosen=int(chosen),
            description=describe_action(ids, int(chosen), pos, is_preview),
            probability=float(probabilities[chosen]),
            legal_actions=count,
            entropy=float(-(probabilities * logits.log_softmax(-1)[0]).sum()),
            top=[
                dict(
                    action=int(a),
                    probability=float(p),
                    description=describe_action(ids, int(a), pos, is_preview),
                )
                for p, a in zip(values, indices)
            ],
        )

    mon_features = part(f, "f", "mon_feats")[0]
    active_ids = part(i, "i", "act_ids")[0]
    public_state = []
    for pos, label in enumerate(
        ("own-left", "own-right", "opponent-left", "opponent-right")
    ):
        index = int(active_ids[pos, 0]) - 1
        public_state.append(
            dict(
                position=label,
                species=NAMES["species"][int(part(i, "i", "mon_ids")[0, index, 0])]
                if index >= 0
                else None,
                hp_fraction=float(mon_features[index, 7]) if index >= 0 else None,
            )
        )
    packed = np.packbits(mask[0].cpu().numpy().astype(np.uint8), bitorder="little")
    result = dict(
        public_active=public_state,
        mask_packed_hex=packed.tobytes().hex(),
        mask_bits=mask.shape[-1],
        mask_bitorder="little",
        alternatives="slot-1 probabilities condition on chosen slot-0 action",
    )
    if preview:
        logits = model.preview_logits(mons)
        result["preview"] = summarize(
            logits,
            torch.ones_like(logits, dtype=torch.bool),
            int(actions[0, 0]),
            0,
            True,
        )
        result["joint_probability"] = result["preview"]["probability"]
    else:
        first, second = (int(x) for x in actions[0])
        logits0, legal0 = model._masked(
            model.position_logits(mons, acts, moves, 0), mask[:, :N_ACTIONS]
        )
        mask1 = mask[:, N_ACTIONS:].reshape(-1, N_ACTIONS, N_ACTIONS)[:, first]
        logits1, legal1 = model._masked(
            model.position_logits(mons, acts, moves, 1, actions[:, 0]), mask1
        )
        result["slot0"] = summarize(logits0, legal0, first, 0)
        result["slot1_given_slot0"] = summarize(logits1, legal1, second, 1)
        result["joint_probability"] = (
            result["slot0"]["probability"] * result["slot1_given_slot0"]["probability"]
        )
    if history is not None:
        result["history"] = [x[0].cpu().tolist() for x in history]
    return result
