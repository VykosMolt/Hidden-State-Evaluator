"""Data locations and pinned revisions for every script in this package.

Locations come from environment variables only (template: ../config.example.sh); no script contains an absolute
path. A script asks for a location with env("RPE_..."), which stops with a message if the variable is unset.
"""
import os
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RPE_RESULTS", PKG / "results"))  # where results are read from and written to

HH_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"  # Anthropic/hh-rlhf commit used by every run
THINKING = ("ByteDance/Ouro-2.6B-Thinking", "f1edd81e7ac41355db670500ceaf204e0f73af68")
BASE = ("ByteDance/Ouro-2.6B", "1ed04250da1a9936042725d302e81c8fa2ab5abd")
RLTT_SHARDS = {"model-00001.safetensors": "9f49e55b0d0ea368f2942cfd3fa256836ea7fa9002c6f860ca2d85ac1de252a1",
               "model-00002.safetensors": "97a0d531a2c624dd98e60c0ebc46a7cc0413642f8cba6df7dac8df01a432df19",
               "model-00003.safetensors": "0bf3d0c111c876ccc7f7ebaeb7cabc8dfccffde30acb4cd780a4c364d3679aad"}

VARS = {
    "RPE_RUNS": "rpe_v3_runs/: evaluator per-pair shards (A_thinking_1024, A_rltt) and probe features (thinking_left, rltt_left_cloud)",
    "RPE_SPLIT": "the 40,000-pair selection and pair-disjoint split (hh_train_indices_40000.json, hh_*_40000_pair_disjoint_probe.pt) and version 2's probe results",
    "RPE_AUDIT": "version 2's full-test-split evaluator audit (full_hh_antisymmetry.json, full_hh_antisymmetry_rows.csv)",
    "RPE_CKPT_DIR": "version 1's evaluator checkpoints pairwise_epoch{1..5}.pt",
    "RPE_RLTT_MODEL": "local Ouro-RLTT checkpoint directory (weights and the tokenizer used for every backbone)",
}


def env(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"{name} is not set: {VARS.get(name, '')} (see config.example.sh)")
    return Path(v)


def load_hh(split: str):
    from datasets import load_dataset
    return load_dataset("Anthropic/hh-rlhf", split=split, revision=HH_REVISION)


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(env("RPE_RLTT_MODEL")), trust_remote_code=True, local_files_only=True)
