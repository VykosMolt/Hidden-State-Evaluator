"""Jacobian lens over Ouro's recurrent depth.

Ouro runs its shared decoder layers `total_ut_steps` times. jlens's
ActivationRecorder keys activations by the index it hooked, so on the native
forward every recurrent pass overwrites the previous one and only the last pass
survives. Here `model.layers` is a flat list of 192 virtual blocks indexed
`ut * 48 + layer`; each is a LoopTap that registers its hook on the shared
physical module but only fires when Ouro's own `current_ut` kwarg matches.
Everything else in jlens (fit / apply / merge) runs unmodified over virtual
indices.

Indexing: `ut` is Ouro's 0-based recurrent step (== `exit_at_step`), `layer`
is the 0-based physical decoder index. Human-facing "loop k" == ut k-1.
"""

from __future__ import annotations

from pathlib import Path

import torch
import transformers

import jlens

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OURO_REVISION = "1ed04250da1a9936042725d302e81c8fa2ab5abd"
OURO_SNAPSHOT = (
    PROJECT_ROOT / "artifacts" / "hf_cache" / "hub" / "models--ByteDance--Ouro-2.6B"
    / "snapshots" / OURO_REVISION
)


class LoopTap:
    """Hook target for one (recurrent step, physical layer) pair."""

    def __init__(self, block: torch.nn.Module, ut: int) -> None:
        self.block = block
        self.ut = ut

    def register_forward_hook(self, hook):
        def filtered(module, args, kwargs, output):
            if kwargs["current_ut"] == self.ut:
                hook(module, args, output)

        return self.block.register_forward_hook(filtered, with_kwargs=True)


class OuroLensModel(jlens.HFLensModel):
    def __init__(self, hf_model, tokenizer) -> None:
        super().__init__(hf_model, tokenizer, force_bos=False)
        self.hf_model = hf_model
        self.blocks = self.layers
        self.n_physical = len(self.blocks)
        self.n_ut = hf_model.model.total_ut_steps
        self.layers = [
            LoopTap(block, ut) for ut in range(self.n_ut) for block in self.blocks
        ]
        self.n_layers = len(self.layers)

    def index(self, ut: int, layer: int) -> int:
        assert 0 <= ut < self.n_ut and 0 <= layer < self.n_physical
        return ut * self.n_physical + layer

    def split(self, virtual: int) -> tuple[int, int]:
        return divmod(virtual, self.n_physical)

    def exit_index(self, ut: int) -> int:
        return self.index(ut, self.n_physical - 1)

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        ids = self.tokenizer(text, truncation=True, max_length=max_length - 1).input_ids
        ids = [self.tokenizer.bos_token_id, *ids]
        return torch.tensor([ids], device=self.input_device)


def load_ouro(path: str | Path = OURO_SNAPSHOT, device: str = "cuda", dtype=torch.bfloat16):
    path = str(path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    return OuroLensModel(hf_model, tokenizer)
