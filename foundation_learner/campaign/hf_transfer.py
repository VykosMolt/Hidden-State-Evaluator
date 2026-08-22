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


def git_blob_sha1(path: str) -> str:
    """The id the Hub assigns a non-LFS file: sha1("blob <size>\\0" + bytes)."""
    import hashlib
    size = os.path.getsize(path)
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode())
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_identity(api, repo_id: str, remote_rel: str,
                    revision: str | None = None) -> dict:
    """What the Hub ACTUALLY holds at ``remote_rel``: the LFS sha256 for an
    LFS object, else the git blob sha1.  Both are computable from local
    bytes, so a transfer can be verified end to end without a second
    upload and without trusting the client's own copy.  ``revision`` pins
    the read to the commit just created: every FL push overwrites the same
    path, so an unpinned read-after-write miss would return the PREVIOUS
    blob and refuse a correct transfer."""
    infos = api.get_paths_info(repo_id, [remote_rel], repo_type="model",
                               revision=revision)
    if not infos:
        raise RuntimeError(f"{remote_rel} is absent from {repo_id} after "
                           f"the transfer")
    info = infos[0]
    lfs = getattr(info, "lfs", None)
    if lfs is not None and getattr(lfs, "sha256", None):
        return {"kind": "lfs", "sha256": lfs.sha256}
    return {"kind": "blob", "sha1": getattr(info, "blob_id", None)}


def verify_against_remote(api, repo_id: str, remote_rel: str,
                          local: str, revision: str | None = None) -> dict:
    ident = remote_identity(api, repo_id, remote_rel, revision=revision)
    if ident["kind"] == "lfs":
        ok = ident["sha256"] == sha256_file(local)
    else:
        ok = ident["sha1"] == git_blob_sha1(local)
    if not ok:
        raise RuntimeError(
            f"{remote_rel}: the Hub's copy does not match the local bytes "
            f"({ident})")
    return {**ident, "remote_verified": True}


def do_push(repo_id: str, local: str, remote_rel: str) -> dict:
    _assert_online_capable()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    digest = sha256_file(local)
    commit = api.upload_file(path_or_fileobj=local, path_in_repo=remote_rel,
                             repo_id=repo_id, repo_type="model")
    remote = verify_against_remote(api, repo_id, remote_rel, local,
                                   revision=getattr(commit, "oid", None))
    return {"remote": remote_rel, "sha256": digest, **remote}


def do_fetch(repo_id: str, remote_rel: str, local: str) -> dict:
    _assert_online_capable()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    # resolve the branch head ONCE and pin both the download and the
    # verification to it, so a concurrent push cannot make the check
    # compare bytes from one commit against identity from another
    revision = api.repo_info(repo_id, repo_type="model").sha
    if not revision:
        raise RuntimeError(f"{repo_id}: repo_info returned no commit sha; "
                           f"refusing an unpinned fetch")
    got = hf_hub_download(repo_id=repo_id, filename=remote_rel,
                          revision=revision,
                          token=os.environ.get("HF_TOKEN"))
    os.makedirs(os.path.dirname(os.path.abspath(local)) or ".", exist_ok=True)
    tmp = local + ".tmp"
    shutil.copyfile(got, tmp)
    digest = sha256_file(tmp)
    # verify the bytes we are about to publish against what the Hub says it
    # holds, BEFORE os.replace makes them the local truth
    remote = verify_against_remote(api, repo_id, remote_rel, tmp,
                                   revision=revision)
    os.replace(tmp, local)
    return {"local": local, "sha256": digest, **remote}


#: Dedicated prefix for the write probe, so it can never collide with a
#: journal or checkpoint key.
PREFLIGHT_PREFIX = ".preflight"


def do_scope(repo_id: str, mode: str) -> dict:
    """Prove the token can do what the session is about to depend on.

    One HF_TOKEN covers both the pregen read and the durable-mirror write.
    A read-only token pulls the corpus, trains for hours, and loses every
    journal and checkpoint push — under interruptible capacity that means
    eviction destroys the arm.  Read is proven by auth_check; write can only
    be proven by writing, so the probe uploads a few bytes and deletes them.
    """
    _assert_online_capable()
    from uuid import uuid4
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    who = api.whoami()
    auth = (who.get("auth") or {}).get("accessToken") or {}
    out = {"repo": repo_id, "mode": mode,
           "identity": who.get("name") or "unknown",
           "declared_role": auth.get("role") or "unreported",
           "read": False, "write": False}
    api.auth_check(repo_id, repo_type="model")
    out["read"] = True
    if mode == "write":
        # Sweep probes a previous pod left behind.  Eviction between the
        # upload and the delete strands a .preflight/ entry in the results
        # repository forever, and nothing else ever removes it.
        swept = []
        for name in api.list_repo_files(repo_id):
            if name.startswith(PREFLIGHT_PREFIX + "/"):
                try:
                    api.delete_file(path_in_repo=name, repo_id=repo_id,
                                    repo_type="model")
                    swept.append(name)
                except Exception:  # noqa: BLE001 - best effort cleanup
                    pass
        out["swept_stale_probes"] = swept
        rel = f"{PREFLIGHT_PREFIX}/scope_{uuid4().hex}"
        api.upload_file(path_or_fileobj=b"fl-b200 write-scope probe\n",
                        path_in_repo=rel, repo_id=repo_id, repo_type="model")
        out["write"] = True
        out["probe_path"] = rel
        # finally: the probe must not survive a failure between here and
        # the return, or it becomes the litter this sweep exists to remove.
        try:
            api.delete_file(path_in_repo=rel, repo_id=repo_id,
                            repo_type="model")
        except Exception:  # noqa: BLE001
            out["probe_cleanup"] = "FAILED (swept on the next acquisition)"
    return out


def do_snapshot(repo_id: str, prefix: str, local: str) -> dict:
    """Materialise a whole staged directory (the pregenerated episode tree).

    The ~480 MB pregen corpus is neither in Git nor in the container image,
    so a fresh pod has to pull it before the ladder can start.  Nothing here
    is trusted: fetch_pregen.py re-hashes every shard against SHARD_SUMS.json
    afterwards.
    """
    _assert_online_capable()
    from huggingface_hub import snapshot_download
    patterns = [f"{prefix}/**", prefix] if prefix else None
    got = snapshot_download(repo_id=repo_id, local_dir=local,
                            allow_patterns=patterns,
                            token=os.environ.get("HF_TOKEN"))
    files = []
    for dirpath, _dirs, names in os.walk(got):
        for name in sorted(names):
            files.append(os.path.relpath(os.path.join(dirpath, name), got))
    return {"local": got, "files": sorted(files), "count": len(files)}


def do_upload_folder(repo_id: str, local: str, remote_prefix: str) -> dict:
    """Stage a whole directory (the pregenerated episode corpus).

    Upload only: nothing here is trusted.  The staged tree is proven by
    re-downloading it through the pod's own fetch path and re-hashing every
    shard against SHARD_SUMS.json -- see scripts/stage_pregen_hf.py.
    """
    _assert_online_capable()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.repo_info(repo_id, repo_type="model")
    if not getattr(info, "private", False):
        raise SystemExit(
            f"hf_transfer: {repo_id} is PUBLIC; refusing to stage campaign "
            f"artefacts into a public repository")
    api.upload_folder(folder_path=local, path_in_repo=remote_prefix,
                      repo_id=repo_id, repo_type="model")
    files = []
    for dirpath, _dirs, names in os.walk(local):
        for name in sorted(names):
            files.append(os.path.relpath(os.path.join(dirpath, name), local))
    return {"repo": repo_id, "remote_prefix": remote_prefix,
            "count": len(files)}


def do_list(repo_id: str, prefix: str) -> dict:
    _assert_online_capable()
    from huggingface_hub import HfApi
    files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(repo_id)
    return {"files": sorted(f for f in files if f.startswith(prefix))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command",
                   choices=["push", "fetch", "list", "snapshot", "scope",
                            "upload-folder"])
    p.add_argument("--repo", required=True)
    p.add_argument("--local")
    p.add_argument("--remote-rel")
    p.add_argument("--prefix", default="")
    p.add_argument("--mode", choices=["read", "write"], default="read")
    a = p.parse_args()
    if a.command == "push":
        out = do_push(a.repo, a.local, a.remote_rel)
    elif a.command == "fetch":
        out = do_fetch(a.repo, a.remote_rel, a.local)
    elif a.command == "snapshot":
        out = do_snapshot(a.repo, a.prefix, a.local)
    elif a.command == "scope":
        out = do_scope(a.repo, a.mode)
    elif a.command == "upload-folder":
        out = do_upload_folder(a.repo, a.local, a.prefix)
    else:
        out = do_list(a.repo, a.prefix)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
