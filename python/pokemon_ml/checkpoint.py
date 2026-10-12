"""Versioned model/observation contracts and resumable RNG state."""

import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from .env import LAYOUT, TABLES, NAMES


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    from .features import FEATURE_SCHEMA

    h = hashlib.sha256(
        json.dumps(
            dict(layout=LAYOUT, features=FEATURE_SCHEMA), sort_keys=True
        ).encode()
    )
    h.update(json.dumps(NAMES, sort_keys=True).encode())
    for key in sorted(TABLES):
        a = np.asarray(TABLES[key])
        h.update(key.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return {"version": 1, "sha256": h.hexdigest()}


def verify_contract(saved):
    if "contract" in saved and saved["contract"] != contract():
        raise ValueError(
            "checkpoint observation/dex contract differs; use matching code or start a new run"
        )


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError(
                "CUDA device count changed; exact RNG restoration is unavailable"
            )
        torch.cuda.set_rng_state_all(state["cuda"])
