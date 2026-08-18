#!/usr/bin/env python3
"""Isolated Hugging Face transfer helper for the FL durable mirror.

Deliberately self-contained (no O1 import — isolation contract) and
deliberately a SUBPROCESS.  The pod sets ``HF_HUB_OFFLINE=1`` so that model
and tokenizer loading can never reach the network; huggingface_hub honours
that by mounting an adapter that raises for every request, uploads
included, and it reads the flag into a module constant at import time.  The
same flag that protects the science would therefore disable the durability
mirror that preemptible execution depends on.

So the hub is reachable ONLY here: the supervisor process stays
offline-locked, and this helper is spawned with the offline flags stripped.
It never imports torch or transformers and never loads a model.

Each operation prints one JSON object on stdout.  Filtering happens INSIDE
this helper, so the FL process never even sees non-FL filenames from a
shared repository.  Credentials arrive via the environment, never echoed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

OFFLINE_FLAGS = ("HF_HUB_OFFLINE", "TRANSFERS_OFFLINE", "TRANSFORMERS_OFFLINE")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def child_env(token: str | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in OFFLINE_FLAGS}
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if token:
        env["HF_TOKEN"] = token
    return env


def _assert_online_capable() -> None:
    from huggingface_hub import constants
    if getattr(constants, "HF_HUB_OFFLINE", False) or \
            os.environ.get("HF_HUB_OFFLINE"):
        raise SystemExit(
            "hf_transfer: HF_HUB_OFFLINE is still set in the helper "
            "process; the FL durable mirror cannot operate")


def do_push(repo_id: str, local: str, remote_rel: str) -> dict:
    _assert_online_capable()
    from huggingface_hub import HfApi
    digest = sha256_file(local)
    HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
        path_or_fileobj=local, path_in_repo=remote_rel,
        repo_id=repo_id, repo_type="model")
    return {"remote": remote_rel, "sha256": digest}


def do_fetch(repo_id: str, remote_rel: str, local: str) -> dict:
    _assert_online_capable()
    from huggingface_hub import hf_hub_download
    got = hf_hub_download(repo_id=repo_id, filename=remote_rel,
                          token=os.environ.get("HF_TOKEN"))
    os.makedirs(os.path.dirname(os.path.abspath(local)) or ".", exist_ok=True)
    tmp = local + ".tmp"
    shutil.copyfile(got, tmp)
    digest = sha256_file(tmp)
    os.replace(tmp, local)
    return {"local": local, "sha256": digest}


def do_list(repo_id: str, prefix: str) -> dict:
    _assert_online_capable()
    from huggingface_hub import HfApi
    files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(repo_id)
    return {"files": sorted(f for f in files if f.startswith(prefix))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["push", "fetch", "list"])
    p.add_argument("--repo", required=True)
    p.add_argument("--local")
    p.add_argument("--remote-rel")
    p.add_argument("--prefix", default="")
    a = p.parse_args()
    if a.command == "push":
        out = do_push(a.repo, a.local, a.remote_rel)
    elif a.command == "fetch":
        out = do_fetch(a.repo, a.remote_rel, a.local)
    else:
        out = do_list(a.repo, a.prefix)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
