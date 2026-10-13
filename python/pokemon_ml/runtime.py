"""Execution options, separate from model architecture and checkpoint weights."""

from contextlib import nullcontext
from dataclasses import dataclass, asdict
import torch


@dataclass
class Runtime:
    precision: str = "fp32"
    attention: str = "auto"
    compile: bool = False
    deferred_metrics: bool = False
    fused_adam: bool = False

    def __post_init__(self):
        if self.precision not in ("fp32", "bf16") or self.attention not in (
            "auto",
            "sdpa",
            "math",
            "flash",
        ):
            raise ValueError("unsupported precision/attention option")


def configure(model, options):
    device = next(model.parameters()).device
    if options.precision == "bf16" and (
        device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 8
    ):
        raise ValueError("BF16 experiment requires an Ampere-or-newer CUDA GPU")
    if options.fused_adam and device.type != "cuda":
        raise ValueError("fused Adam experiment requires CUDA")
    if options.attention == "flash" and device.type != "cuda":
        raise ValueError("Flash attention experiment requires CUDA")
    model.runtime = options
    # Preserve the original modules/state_dict. Compile only the tensor encoder.
    # The eager policy heads retain FP32 probability/loss arithmetic.
    model._runtime_encoder = (
        torch.compile(model._encode, dynamic=True, fullgraph=True)
        if options.compile
        else model._encode
    )
    return model


def context(model, device):
    options = getattr(model, "runtime", Runtime())
    if options.attention in ("math", "flash"):
        from torch.nn.attention import sdpa_kernel, SDPBackend

        return sdpa_kernel(
            SDPBackend.MATH
            if options.attention == "math"
            else SDPBackend.FLASH_ATTENTION
        )
    return nullcontext()


def metadata(options):
    return asdict(options)
