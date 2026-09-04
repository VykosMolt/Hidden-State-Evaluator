"""CPU-only checks for the recurrent Jacobian-lens helpers (no model download)."""

import numpy as np
import torch
from torch import nn

from jlens.hooks import ActivationRecorder
from ouro_jlens.evaldata import surface_forms
from ouro_jlens.recurrent import LoopTap


class Block(nn.Module):
    def forward(self, x, current_ut=0):
        return x + 1 + 10 * current_ut


def test_loop_tap_records_each_recurrent_invocation_separately():
    blocks = nn.ModuleList([Block(), Block()])
    taps = [LoopTap(b, ut) for ut in range(3) for b in blocks]
    with ActivationRecorder(taps, at=range(6), start_graph_at=0) as rec:
        x = torch.zeros(1, 2)
        for ut in range(3):
            for b in blocks:
                x = b(x, current_ut=ut)
    assert sorted(rec.activations) == list(range(6))
    values = [rec.activations[v][0, 0].item() for v in range(6)]
    assert values == [1, 2, 13, 24, 45, 66]
    assert rec.activations[0].requires_grad and rec.activations[0].is_leaf
    assert all(len(b._forward_hooks) == 0 for b in blocks)


def test_loop_tap_hook_fires_only_for_its_loop():
    block = Block()
    seen = []
    handle = LoopTap(block, 1).register_forward_hook(lambda m, a, o: seen.append(o.item()))
    for ut in range(3):
        block(torch.zeros(1), current_ut=ut)
    handle.remove()
    assert seen == [11.0]


def test_surface_forms_cover_digit_word_and_operator_forms():
    assert {"12", "twelve"} <= set(surface_forms("12"))
    assert {"five", "5"} <= set(surface_forms("five"))
    assert {"*", "times", "multiplication"} <= set(surface_forms("multiplication"))


def _fake_eval(tmp_path, n=30):
    """Two-name task; item i owns name 0 (rank 0 where planted, else 500), name 1 is the control."""
    import json

    from ouro_jlens.analyze import Eval

    names = ["alpha", "beta"]
    allrank = np.full((n, 128, 192), -1, np.int32)
    allrank[:, :2] = 500
    allrank[:, 0, 3 * 48 + 20] = 0                   # own name readable at (loop 4, layer 20)
    allrank[:, 1, 3 * 48 + 20] = 3                   # control also "hits" there (hit@10) but ranks below own
    xloop = np.full((n, 128, 4, 4, 48), -1, np.int32)
    xloop[:, :2] = 500
    xloop[:, 0, 1:, 1:, 20] = 0                      # loops 2-4 transfer among themselves
    xloop[:, 0, 0, 0, 20] = 0                        # loop 1 only matches itself
    xloop[:, 1, 0, 0, 20] = 0                        # ...but so does the control there -> no excess
    np.savez(tmp_path / "arrays.npz", jlens_exit3_allrank=allrank, logitlens_allrank=allrank, xloop_allrank=xloop,
             exit_top1=np.zeros((n, 4), np.int64), jlens_exit3_top1=np.zeros((n, 192), np.int64))
    items = [{"name": f"i{i}", "task": "multihop", "intermediates": ["alpha"], "own_index": [0, -1, -1],
              "scorable": [True], "leaked": [False], "correct": True} for i in range(n)]
    (tmp_path / "items.json").write_text(json.dumps(items))
    (tmp_path / "task_names.json").write_text(json.dumps({"multihop": names}))
    return Eval(tmp_path)


def test_scores_separate_own_hits_from_control_prior(tmp_path):
    from ouro_jlens.analyze import loc_maps

    ev = _fake_eval(tmp_path)
    m = loc_maps(ev.scores(ev.arrays["jlens_exit3_allrank"], ev.slot_mask["multihop"]))
    assert m["hit"][3, 20] == 1 and m["control"][3, 20] == 1 and m["excess"][3, 20] == 0
    assert m["cand_top1"][3, 20] == 1 and m["cand_top1"].sum() == 1 and m["hit"].sum() == 1


def test_cross_loop_summary_recovers_planted_structure(tmp_path):
    from ouro_jlens.analyze import cross_loop_summary

    ev = _fake_eval(tmp_path)
    res = cross_loop_summary(ev, ev.slot_mask["multihop"])
    M = np.array(res["excess_hit10_fit_by_state"])
    assert M[0, 0] == 0 and M[1:, 1:].min() == 1 and M[0, 1:].max() == 0 and M[1:, 0].max() == 0
    assert res["h1_loop1_minus_later_offdiag"] == -1.0 and res["n_items"] == 30


def test_readout_context_reads_before_target_first_token():
    from ouro_jlens.evaldata import readout_context

    vocab = {"Fact": 1, " is": 2, " ": 3, " Atlantic": 4, " =": 5, "2": 6, "0": 7}

    def encode(s):  # greedy longest-match toy tokenizer
        out, i = [], 0
        while i < len(s):
            tok = max((t for t in vocab if s.startswith(t, i)), key=len)
            out.append(vocab[tok]); i += len(tok)
        return out

    ids, dropped = readout_context(encode, "Fact is ", "Atlantic")
    assert ids == [1, 2] and dropped == 1          # trailing space merged into " Atlantic"
    ids, dropped = readout_context(encode, "Fact = ", "20")
    assert ids == [1, 5, 3] and dropped == 0       # digits follow the lone space token
