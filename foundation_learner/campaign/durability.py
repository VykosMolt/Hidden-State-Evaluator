"""Continuous off-pod durability for the FL session under preemption.

INTERRUPTIBLE accelerator capacity can be evicted with ~5s notice, and the
session pod carries container disk only — anything not already off-pod is
lost.  This module mirrors the FL session's durable state continuously:

  * the supervisor journal (the resume source of truth) after every append;
  * every atomic training checkpoint (payload + manifest, manifest LAST so
    a half-pushed checkpoint is never referenced);
  * O1-close receipts and other session receipts as they are written.

On a fresh pod after eviction, ``restore_all`` repopulates the FL output
tree from the mirror BEFORE the supervisor reads its journal, after which
the EXISTING resume machinery proceeds exactly as it would after a
same-pod restart: completed states skip, training resumes from the last
valid atomic checkpoint, and work performed after the last durable
checkpoint is simply rerun.  No hyperparameter may change on resume — the
checkpoint manifest binds the arm configuration hash and the trainer
refuses a mismatch.

Restore NEVER overwrites an existing local file (local atomic writes win)
and verifies every fetched object against the Hub's identity for it; a corrupt mirrored file is
quarantined, never restored.  This module deliberately imports nothing
from the O1 package (isolation contract): the store pattern is
re-implemented, not shared.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time

_PREFIX = "fl_durable"


class DurabilityError(RuntimeError):
    pass


def parse_hf_destination(destination: str) -> tuple[str, str]:
    """``hf://ns/repo[/sub/path]`` -> (``ns/repo``, sub-path or ``""``).

    The sub-path is a documented session scope.  Silently discarding it
    would let two campaigns that share a repo (``hf://ns/repo/fl`` vs
    ``hf://ns/repo/o1``) overwrite each other's durable state.
    """
    if not destination.startswith("hf://"):
        raise DurabilityError(f"not an hf:// destination: {destination!r}")
    parts = destination[len("hf://"):].strip("/").split("/")
    if len(parts) < 2 or not all(parts[:2]):
        raise DurabilityError(f"malformed destination {destination!r}")
    return "/".join(parts[:2]), "/".join(parts[2:]).strip("/")


def _safe_session_id(session_id: str) -> str:
    text = str(session_id).strip()
    if (not text or text in (".", "..") or "/" in text or "\\" in text
            or text != os.path.basename(text)):
        raise DurabilityError(f"invalid session_id {session_id!r}")
    return text


def _contained(root: str, path: str) -> bool:
    root_abs = os.path.abspath(root)
    path_abs = os.path.abspath(path)
    return path_abs == root_abs or path_abs.startswith(root_abs + os.sep)


def _sha256_file(path: str, guard=None) -> str:
    if guard is not None:
        path = guard(path, "read")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _LocalStore:
    def __init__(self, root: str, guard=None):
        self.guard = guard or (lambda path, mode="read": path)
        self.root = os.path.abspath(self.guard(root, "write"))
        os.makedirs(self.root, exist_ok=True)

    def _p(self, rel: str) -> str:
        p = os.path.abspath(os.path.join(self.root, rel))
        if not p.startswith(self.root + os.sep):
            raise DurabilityError(f"path escape refused: {rel!r}")
        return p

    def push(self, local: str, rel: str) -> None:
        local = self.guard(local, "read")
        dest = self.guard(self._p(rel), "write")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".tmp"
        shutil.copyfile(local, tmp)
        if _sha256_file(tmp, self.guard) != _sha256_file(local, self.guard):
            os.remove(tmp)
            raise DurabilityError(f"push corruption for {rel}")
        os.replace(tmp, dest)

    def fetch(self, rel: str, local: str) -> None:
        src = self.guard(self._p(rel), "read")
        local = self.guard(local, "write")
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        tmp = local + ".tmp"
        shutil.copyfile(src, tmp)
        if _sha256_file(tmp, self.guard) != _sha256_file(src, self.guard):
            os.remove(tmp)
            raise DurabilityError(f"fetch corruption for {rel}")
        os.replace(tmp, local)

    def list_all(self) -> list[str]:
        out = []
        for dirpath, _dirs, files in os.walk(self.root):
            for name in files:
                out.append(os.path.relpath(os.path.join(dirpath, name),
                                           self.root))
        return sorted(out)


class _HfStore:
    """Private Hugging Face repo store (token via HF_TOKEN).

    Every hub call runs in the isolated ``campaign.hf_transfer`` subprocess
    so the supervisor process itself stays offline-locked (see that module),
    and every transfer is verified against the Hub's own blob/LFS identity end to end — an unverified copy
    would let a silently-corrupted mirror be restored as if it were sound.
    Prefix filtering happens inside the helper, so this process never sees
    non-FL filenames from a shared repository.
    """

    def __init__(self, repo_id: str, guard=None, runner=None,
                 timeout: float = 3600.0, list_prefix: str = _PREFIX):
        self.guard = guard or (lambda path, mode="read": path)
        self.repo_id = repo_id
        self.token = os.environ.get("HF_TOKEN")
        self.timeout = timeout
        self.list_prefix = list_prefix
        self._runner = runner or self._spawn

    def _spawn(self, args: list[str]) -> dict:
        import subprocess
        import sys

        from .hf_transfer import child_env
        proc = subprocess.run(
            [sys.executable, "-m", "foundation_learner.campaign.hf_transfer",
             *args],
            capture_output=True, text=True, timeout=self.timeout,
            env=child_env(self.token))
        if proc.returncode != 0:
            detail = (proc.stdout + proc.stderr)[-400:]
            if self.token:
                detail = detail.replace(self.token, "[REDACTED]")
            raise DurabilityError(f"hf {args[0]} failed: {detail}")
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            raise DurabilityError(
                f"hf {args[0]} produced no result object") from None

    def push(self, local: str, rel: str) -> None:
        local = self.guard(local, "read")
        out = self._runner(["push", "--repo", self.repo_id,
                            "--local", local, "--remote-rel", rel])
        if out.get("sha256") != _sha256_file(local):
            raise DurabilityError(f"push digest disagreement for {rel}")
        if not out.get("remote_verified"):
            raise DurabilityError(
                f"push of {rel} was not verified against the Hub's copy")

    def fetch(self, rel: str, local: str) -> None:
        local = self.guard(local, "write")
        out = self._runner(["fetch", "--repo", self.repo_id,
                            "--remote-rel", rel, "--local", local])
        if out.get("sha256") != _sha256_file(local):
            raise DurabilityError(f"fetch digest disagreement for {rel}")
        if not out.get("remote_verified"):
            raise DurabilityError(
                f"fetch of {rel} was not verified against the Hub's copy")

    def list_all(self) -> list[str]:
        out = self._runner(["list", "--repo", self.repo_id,
                            "--prefix", self.list_prefix])
        return list(out.get("files") or [])


def _store_for(destination: str, guard=None, *, list_prefix: str | None = None):
    if destination.startswith("hf://"):
        repo_id, _subpath = parse_hf_destination(destination)
        prefix = _PREFIX if list_prefix is None else list_prefix
        return _HfStore(repo_id, guard=guard, list_prefix=prefix)
    return _LocalStore(destination, guard=guard)


class FlDurableMirror:
    """Mirrors files under ``run_root`` to the durable destination."""

    def __init__(self, destination: str, run_root: str, on_event=None,
                 guard=None, session_id: str | None = None):
        # guard: the campaign isolation guard (o1_isolation.IsolationGuard
        # .guard); every local path this mirror touches is routed through it
        self.guard = guard or (lambda path, mode="read": path)
        self.session_id = (_safe_session_id(session_id)
                           if session_id else None)
        if destination.startswith("hf://"):
            _repo, self.remote_subpath = parse_hf_destination(destination)
        else:
            self.remote_subpath = ""
        self.run_root = os.path.abspath(run_root)
        self.on_event = on_event or (lambda *a, **k: None)
        self._last_push_at = {}
        self._pending = {}
        self.consecutive_failures = 0
        self.degraded = False
        self.store = _store_for(destination, guard=self.guard,
                                list_prefix=self._scope_prefix())

    def _scope_prefix(self) -> str:
        """Remote key prefix: optional dest sub-path + fl_durable + session."""
        parts: list[str] = []
        if self.remote_subpath:
            parts.append(self.remote_subpath.strip("/"))
        parts.append(_PREFIX)
        if self.session_id:
            parts.append(self.session_id)
        return "/".join(parts)

    def _rel(self, path: str) -> str:
        ap = os.path.abspath(path)
        if not _contained(self.run_root, ap) or ap == self.run_root:
            raise DurabilityError(
                f"{path!r} is outside the mirrored run root")
        return f"{self._scope_prefix()}/" + os.path.relpath(ap, self.run_root)

    #: Throttle for high-frequency files (the ladder journal is appended
    #: on every heartbeat, i.e. every 25 optimizer steps).  Each push is a
    #: subprocess + one HF commit; thousands per session burned paid
    #: wall-clock inside the training loop and invited hub rate limiting.
    JOURNAL_PUSH_INTERVAL_SECONDS = 60.0
    #: consecutive failures after which the mirror is reported DEGRADED
    DEGRADED_AFTER_FAILURES = 5

    _last_push_at: dict  # rel -> monotonic
    _pending: dict       # rel -> local path awaiting a throttled push
    consecutive_failures: int = 0
    degraded: bool = False

    def push_file(self, path: str) -> None:
        """Best-effort single-file mirror; failure is reported loudly but
        never destroys local state (it widens the eviction-loss window).
        Consecutive failures are counted and the mirror reports itself
        DEGRADED once, so a dead mirror cannot look durable in the journal."""
        try:
            self.store.push(path, self._rel(path))
        except Exception as exc:  # noqa: BLE001
            self._note_failure(path, exc)
            return
        self._note_success(path)

    def _note_failure(self, path: str, exc: BaseException) -> None:
        self.consecutive_failures += 1
        self.on_event("FL_DURABILITY_PUSH_FAILED",
                      path=os.path.basename(path), error=str(exc)[:200],
                      consecutive_failures=self.consecutive_failures)
        if (self.consecutive_failures >= self.DEGRADED_AFTER_FAILURES
                and not self.degraded):
            self.degraded = True
            self.on_event("FL_DURABILITY_DEGRADED",
                          consecutive_failures=self.consecutive_failures)

    def _note_success(self, path: str) -> None:
        if self.consecutive_failures:
            self.on_event("FL_DURABILITY_RECOVERED",
                          after_failures=self.consecutive_failures)
        self.consecutive_failures = 0
        self.degraded = False
        self._last_push_at[self._rel(path)] = time.monotonic()

    def push_throttled(self, path: str, *, force: bool = False) -> bool:
        """Push ``path`` unless it was pushed less than
        ``JOURNAL_PUSH_INTERVAL_SECONDS`` ago; ``force`` pushes now.  A
        skipped push is remembered so ``flush()`` can complete it.  Returns
        True when a push was attempted."""
        rel = self._rel(path)
        last = self._last_push_at.get(rel)
        if (not force and last is not None
                and time.monotonic() - last < self.JOURNAL_PUSH_INTERVAL_SECONDS):
            self._pending[rel] = path
            return False
        self._pending.pop(rel, None)
        self.push_file(path)
        return True

    def flush(self) -> int:
        """Push everything a throttled call deferred."""
        pending = list(self._pending.values())
        self._pending.clear()
        for path in pending:
            if os.path.isfile(path):
                self.push_file(path)
        return len(pending)

    def push_file_strict(self, path: str) -> None:
        """Mirror one file and RAISE on failure.

        For the records whose durability is itself an invariant (the sealed
        opening ledger and the sealed results): a swallowed push there is
        how a replacement pod could be granted a second opening.
        """
        try:
            self.store.push(path, self._rel(path))
        except Exception as exc:
            self._note_failure(path, exc)
            raise
        self._note_success(path)
        self.on_event("FL_DURABLE_STRICT_PUSHED", path=os.path.basename(path))

    def push_checkpoint(self, payload_path: str, manifest_path: str) -> None:
        """Checkpoint mirror: payload FIRST, manifest LAST — a checkpoint
        whose manifest is durable is guaranteed completely durable."""
        self.store.push(payload_path, self._rel(payload_path))
        self.store.push(manifest_path, self._rel(manifest_path))
        self.on_event("FL_CHECKPOINT_MIRRORED",
                      manifest=os.path.basename(manifest_path))

    def restore_all(self) -> dict:
        """Repopulate the run root from the mirror.  Existing local files
        are never overwritten; a corrupt fetch is quarantined."""
        restored, skipped, corrupt = 0, 0, 0
        quarantine = os.path.join(self.run_root, "durability_quarantine")
        prefix = self._scope_prefix() + "/"
        for rel in self.store.list_all():
            if not rel.startswith(prefix):
                continue
            local_rel = rel[len(prefix):]
            local = os.path.abspath(os.path.join(self.run_root, local_rel))
            if not _contained(self.run_root, local) or local == self.run_root:
                self.on_event("FL_DURABILITY_RESTORE_ESCAPE", path=rel)
                raise DurabilityError(
                    f"path escape refused during restore: {rel!r}")
            if os.path.exists(local):
                skipped += 1
                continue
            tmp = local + ".restoring"
            try:
                self.store.fetch(rel, tmp)
                if not _contained(self.run_root, os.path.abspath(tmp)):
                    raise DurabilityError(
                        f"path escape refused during restore: {rel!r}")
                os.replace(tmp, local)
                restored += 1
            except Exception as exc:  # noqa: BLE001
                corrupt += 1
                os.makedirs(quarantine, exist_ok=True)
                if os.path.exists(tmp) and _contained(self.run_root,
                                                      os.path.abspath(tmp)):
                    os.replace(tmp, os.path.join(
                        quarantine, os.path.basename(local) + ".corrupt"))
                self.on_event("FL_DURABILITY_RESTORE_CORRUPT",
                              path=os.path.basename(local),
                              error=str(exc)[:200])
        report = {"restored": restored, "skipped_existing": skipped,
                  "corrupt_quarantined": corrupt}
        self.on_event("FL_DURABILITY_RESTORED", **report)
        return report


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["restore"])
    p.add_argument("--destination", required=True)
    p.add_argument("--run-root", required=True)
    a = p.parse_args()
    mirror = FlDurableMirror(a.destination, a.run_root)
    print(json.dumps(mirror.restore_all()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
