"""PEER1: causal regression test for the recurrent virtual index map.

`validate.py` milestone 3 checks the index map by reading the same `current_ut`
kwarg that `LoopTap` keys its recording on, so a consistent mis-keying would pass
it. This test instead *intervenes*: it perturbs the output of physical layer L on
recurrent step UP only, then asks which recorded virtual slots changed. If
`index(ut, layer) == ut * n_physical + layer` is right, exactly the slots at or
after `UP * 48 + L` change and every earlier slot is bit-identical.

Needs the GPU and the local Ouro snapshot; skipped otherwise (~20 s).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

CASES = [(0, 5), (1, 20), (2, 33), (3, 0), (2, 47)]


@pytest.fixture(scope="module")
def ouro():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from ouro_jlens.recurrent import OURO_SNAPSHOT, load_ouro

    if not (Path(OURO_SNAPSHOT) / "model.safetensors").exists():
        pytest.skip(f"no local Ouro snapshot at {OURO_SNAPSHOT}")
    return load_ouro()


def _record_all(m, ids):
    from jlens.hooks import ActivationRecorder

    with torch.no_grad(), ActivationRecorder(m.layers, at=range(m.n_layers)) as rec:
        m.forward(ids)
        return {v: rec.activations[v][0].clone() for v in range(m.n_layers)}


@pytest.mark.parametrize(("ut", "layer"), CASES)
def test_perturbing_one_loop_and_layer_changes_exactly_the_slots_at_or_after_it(ouro, ut, layer):
    m = ouro
    ids = m.encode("The capital of the country whose currency is the yen is")
    base = _record_all(m, ids)

    bump = torch.zeros(m.d_model, device=m.input_device, dtype=torch.bfloat16)
    bump[7] = 50.0

    def patch(module, args, kwargs, output):
        return output + bump if kwargs["current_ut"] == ut else output

    handle = m.blocks[layer].register_forward_hook(patch, with_kwargs=True)
    try:
        perturbed = _record_all(m, ids)
    finally:
        handle.remove()

    changed = [v for v in range(m.n_layers) if not torch.equal(base[v], perturbed[v])]
    first = m.index(ut, layer)
    assert first == ut * m.n_physical + layer
    assert changed == list(range(first, m.n_layers)), (
        f"perturbing (ut={ut}, layer={layer}) should change virtual slots "
        f"{first}..{m.n_layers - 1} and nothing earlier; changed={changed[:5]}..."
    )
