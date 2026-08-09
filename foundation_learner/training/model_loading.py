"""Frozen-backbone loading recipe and identity binding (contract §1, §22, §23).

The backbone is FROZEN: exactly one checkpoint, exactly one loading recipe,
``transformers==4.54.1`` asserted hard.  Every training arm starts from a fresh
load of this identical checkpoint; the checkpoint artifact is never modified.

Hash conventions
----------------
Contract §3 asks for a single implementation of the O1 hash conventions in
``ecology/base.py``.  Contract §22 simultaneously requires that FL never imports
sealed O1 modules and re-implements the conventions identically ("23 lines of
stdlib").  The training package therefore carries its own self-contained copy
here (identical algorithm, pinned by the frozen §1 test vectors) instead of
depending on another worker's module at import time.  A regression test
cross-checks this implementation against ``ecology.base`` whenever that module
is importable, so a divergence cannot go unnoticed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping

# --------------------------------------------------------------------------
# Frozen binding (contract §1)
# --------------------------------------------------------------------------

FROZEN_CHECKPOINT_DIR = "/home/moloch/ouro_project/models/ouro_rltt_local"
FROZEN_CHECKPOINT_TREE_SHA256 = (
    "a701f7a75300ddf57098572fef3894bef59d5179580ec7eae7cd561a36056889"
)
REQUIRED_TRANSFORMERS_VERSION = "4.54.1"

FROZEN_ARCHITECTURE = "OuroForCausalLM"
FROZEN_MODEL_TYPE = "ouro"
FROZEN_NUM_HIDDEN_LAYERS = 48
FROZEN_TOTAL_UT_STEPS = 4
FROZEN_HIDDEN_SIZE = 2048
FROZEN_NUM_ATTENTION_HEADS = 16
FROZEN_NUM_KEY_VALUE_HEADS = 16
FROZEN_INTERMEDIATE_SIZE = 5632
FROZEN_VOCAB_SIZE = 49152
FROZEN_MAX_POSITION_EMBEDDINGS = 65536
FROZEN_EARLY_EXIT_THRESHOLD = 1.0
FROZEN_RLTT_LOGPROB_CHUNK_SIZE = 2048
FROZEN_RLTT_LOOP_LEVEL_CHECKPOINTING = True

#: Per-file SHA-256 of the frozen checkpoint's small (non-weight) files (§1).
#: The three ``model-0000?.safetensors`` shards are covered by the tree hash.
FROZEN_FILE_SHA256: dict[str, str] = {
    "config.json": "7d6764dbc8210d023c8d83da4620910808ac5a450532b15550e57d1ef0e4f741",
    "tokenizer.json": "fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d",
    "tokenizer_config.json": "7936619f224ec48539a38f8d7dbc64b3c1ba397f4c4b753397d52116cf71dcd8",
    "vocab.json": "7b9de3f47796abf8d00ab96be299fea0dc9afdf1827f34e7e0b9fb44593efe5c",
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "special_tokens_map.json": "aadabc9bd7e3f4738bc1160ef0aa932ba09401c1f11271bd269a21ba987b353b",
    "configuration_ouro.py": "4c7c6138715351f7b673eed4a8e7553ccccb8a1f1ab93e82e9773ad96c6ee7d6",
    "modeling_ouro.py": "bcd27ff6a18578feaec168695d70dc76509c57e4d4c377347c9d05d6266d9e82",
    "chat_template.jinja": "609d03c963c8d8d0519c9212df62280bc6dd561013259c3b6cd0ee43f7910f58",
    "model.safetensors.index.json": (
        "e1842668e1ba1568364a4ae7227a5ab80ab5403c95baba37b92205d2cb22a001"
    ),
}

MODEL_IDENTITY_SCHEMA = "flb200.model_identity.v1"


class FrozenBindingError(RuntimeError):
    """Raised when the runtime or checkpoint violates the §1/§22 binding."""


# --------------------------------------------------------------------------
# Hash / determinism conventions (contract §3, identical to O1)
# --------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def domain_sha256(domain: str, obj: Any) -> str:
    return hashlib.sha256(
        domain.encode("utf-8") + b"\0" + canonical_json(obj).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(path: str) -> str:
    if os.path.isfile(path):
        return sha256_file(path)
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(bytes.fromhex(sha256_file(full)))
            h.update(b"\n")
    return h.hexdigest()


def derive_seed(root_seed: int, *tags: Any) -> int:
    """Contract §3 substream seed derivation."""
    return int(domain_sha256("FL_V0_SEED", [int(root_seed), *tags])[:16], 16) % 2**63


# --------------------------------------------------------------------------
# Environment / config digests
# --------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


_CONFIG_DIGEST_EXCLUDED = ("_name_or_path", "transformers_version", "name_or_path")


def config_digest(config: Any) -> str:
    """Domain-separated digest of a model config (path/version independent)."""
    raw = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    payload = {
        k: _jsonable(v) for k, v in raw.items() if k not in _CONFIG_DIGEST_EXCLUDED
    }
    return domain_sha256("FL_V0_MODEL_CONFIG", payload)


def environment_digest() -> dict[str, Any]:
    import torch  # local import: keeps hash helpers usable without torch
    import transformers

    try:
        import numpy

        numpy_version = numpy.__version__
    except Exception:  # pragma: no cover - numpy is a declared dependency
        numpy_version = None
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy_version,
        "cuda": getattr(torch.version, "cuda", None),
        "cudnn": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
    }
    env["digest"] = domain_sha256("FL_V0_ENVIRONMENT", _jsonable(env))
    return env


# --------------------------------------------------------------------------
# Verification helpers
# --------------------------------------------------------------------------


def assert_transformers_version() -> str:
    """Hard fail unless transformers is exactly the frozen version (§1/§22)."""
    import transformers

    version = transformers.__version__
    if version != REQUIRED_TRANSFORMERS_VERSION:
        raise FrozenBindingError(
            "transformers version mismatch: required exactly "
            f"{REQUIRED_TRANSFORMERS_VERSION!r}, found {version!r}. "
            "The frozen backbone binding forbids running on any other version."
        )
    return version


def verify_frozen_files(checkpoint_dir: str) -> dict[str, str]:
    """Verify the per-file SHA-256 table from §1 (small files only)."""
    mismatches = []
    seen: dict[str, str] = {}
    for name, expected in sorted(FROZEN_FILE_SHA256.items()):
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isfile(path):
            mismatches.append(f"{name}: MISSING")
            continue
        got = sha256_file(path)
        seen[name] = got
        if got != expected:
            mismatches.append(f"{name}: expected {expected} got {got}")
    if mismatches:
        raise FrozenBindingError(
            "frozen checkpoint file verification failed for "
            f"{checkpoint_dir!r}: " + "; ".join(mismatches)
        )
    return seen


def verify_checkpoint_tree(checkpoint_dir: str) -> str:
    """Verify the whole checkpoint tree against the frozen §1 tree hash."""
    got = sha256_tree(checkpoint_dir)
    if got != FROZEN_CHECKPOINT_TREE_SHA256:
        raise FrozenBindingError(
            f"checkpoint tree hash mismatch for {checkpoint_dir!r}: expected "
            f"{FROZEN_CHECKPOINT_TREE_SHA256} got {got}"
        )
    return got


# --------------------------------------------------------------------------
# ModelBundle
# --------------------------------------------------------------------------


@dataclass
class ModelBundle:
    """Contract §23: ``.model``, ``.tokenizer``, ``.device``, ``.identity``."""

    model: Any
    tokenizer: Any
    device: str
    identity: dict[str, Any] = field(default_factory=dict)

    @property
    def scientific(self) -> bool:
        return bool(self.identity.get("scientific", False))

    @property
    def identity_hash(self) -> str:
        return str(self.identity["identity_hash"])

    def assert_scientific(self) -> None:
        if not self.scientific:
            raise FrozenBindingError(
                "this bundle is marked NONSCIENTIFIC "
                f"({self.identity.get('kind')}) and must not be used on a "
                "scientific path"
            )


def build_identity(
    *,
    kind: str,
    scientific: bool,
    checkpoint_dir: str | None,
    tree_sha256: str | None,
    tree_hash_verified: bool,
    config: Any,
    dtype: str,
    device: str,
    extra: Mapping[str, Any] | None = None,
    core_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the identity dict recorded in every checkpoint and ledger.

    ``core_extra`` adds fields that PARTICIPATE in ``identity_hash`` (e.g. the
    tiny model's init seed, which determines its starting weights); ``extra``
    adds descriptive fields that do not.
    """
    import torch
    import transformers

    digest = config_digest(config)
    architectures = list(getattr(config, "architectures", None) or [])
    core = {
        "architecture": architectures[0] if architectures else None,
        "model_type": getattr(config, "model_type", None),
        "checkpoint_tree_sha256": tree_sha256,
        "config_digest": digest,
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "total_ut_steps": getattr(config, "total_ut_steps", None),
        "early_exit_threshold": getattr(config, "early_exit_threshold", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "dtype": dtype,
        "kind": kind,
    }
    if core_extra:
        core.update({str(k): _jsonable(v) for k, v in core_extra.items()})
    identity = {
        "schema": MODEL_IDENTITY_SCHEMA,
        "kind": kind,
        "scientific": bool(scientific),
        "checkpoint_dir": os.path.realpath(checkpoint_dir) if checkpoint_dir else None,
        "tree_hash_verified": bool(tree_hash_verified),
        "device": device,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        **core,
        "identity_hash": domain_sha256("FL_V0_BASE_IDENTITY", _jsonable(core)),
    }
    if not scientific:
        identity["NONSCIENTIFIC"] = True
        identity["nonscientific_reason"] = (
            "mechanics-test model; never a scientific result path (contract §18/§20)"
        )
    if extra:
        identity.update({str(k): _jsonable(v) for k, v in extra.items()})
    return identity


# --------------------------------------------------------------------------
# The frozen loading recipe (contract §22)
# --------------------------------------------------------------------------


def load_tokenizer(checkpoint_dir: str = FROZEN_CHECKPOINT_DIR):
    """Load the frozen tokenizer only (no weights).  Allowed in unit tests."""
    assert_transformers_version()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        checkpoint_dir, trust_remote_code=True, local_files_only=True
    )


def pin_recurrence_(model) -> None:
    """Pin the frozen UT-step / early-exit configuration (contract §22)."""
    model.config.total_ut_steps = FROZEN_TOTAL_UT_STEPS
    model.config.early_exit_threshold = FROZEN_EARLY_EXIT_THRESHOLD
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "total_ut_steps"):
        inner.total_ut_steps = FROZEN_TOTAL_UT_STEPS
    if hasattr(model, "early_exit_threshold"):
        model.early_exit_threshold = FROZEN_EARLY_EXIT_THRESHOLD


def assert_frozen_architecture(config) -> None:
    problems = []

    def _check(name: str, expected: Any) -> None:
        got = getattr(config, name, None)
        if got != expected:
            problems.append(f"{name}: expected {expected!r} got {got!r}")

    _check("model_type", FROZEN_MODEL_TYPE)
    _check("num_hidden_layers", FROZEN_NUM_HIDDEN_LAYERS)
    _check("hidden_size", FROZEN_HIDDEN_SIZE)
    _check("num_attention_heads", FROZEN_NUM_ATTENTION_HEADS)
    _check("num_key_value_heads", FROZEN_NUM_KEY_VALUE_HEADS)
    _check("intermediate_size", FROZEN_INTERMEDIATE_SIZE)
    _check("vocab_size", FROZEN_VOCAB_SIZE)
    _check("max_position_embeddings", FROZEN_MAX_POSITION_EMBEDDINGS)
    _check("total_ut_steps", FROZEN_TOTAL_UT_STEPS)
    _check("early_exit_threshold", FROZEN_EARLY_EXIT_THRESHOLD)
    _check("rltt_logprob_chunk_size", FROZEN_RLTT_LOGPROB_CHUNK_SIZE)
    _check("rltt_loop_level_checkpointing", FROZEN_RLTT_LOOP_LEVEL_CHECKPOINTING)
    architectures = list(getattr(config, "architectures", []) or [])
    if architectures != [FROZEN_ARCHITECTURE]:
        problems.append(f"architectures: expected ['{FROZEN_ARCHITECTURE}'] got {architectures!r}")
    if problems:
        raise FrozenBindingError(
            "loaded checkpoint violates the frozen §1 backbone binding: "
            + "; ".join(problems)
        )


def load_frozen_backbone(
    checkpoint_dir: str = FROZEN_CHECKPOINT_DIR,
    device: str = "cuda",
    verify_tree_hash: bool = False,
    *,
    verify_files: bool = True,
) -> ModelBundle:
    """Contract §23 entry point; implements the §22 recipe exactly.

    ``verify_tree_hash`` additionally re-hashes the full checkpoint tree
    (~5.3 GB read) and refuses to proceed on mismatch.
    """
    import torch
    from transformers import AutoModelForCausalLM

    assert_transformers_version()
    checkpoint_dir = os.path.realpath(checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        raise FrozenBindingError(f"checkpoint directory not found: {checkpoint_dir!r}")

    if verify_files:
        verify_frozen_files(checkpoint_dir)
    tree_sha256 = verify_checkpoint_tree(checkpoint_dir) if verify_tree_hash else None

    tokenizer = load_tokenizer(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.to(device)
    pin_recurrence_(model)
    assert_frozen_architecture(model.config)

    identity = build_identity(
        kind="FROZEN_BACKBONE",
        scientific=True,
        checkpoint_dir=checkpoint_dir,
        tree_sha256=tree_sha256 or FROZEN_CHECKPOINT_TREE_SHA256,
        tree_hash_verified=bool(verify_tree_hash),
        config=model.config,
        dtype="bfloat16",
        device=device,
        extra={
            "files_verified": bool(verify_files),
            "loading_recipe": "FL_V0_FROZEN_RECIPE_S22",
            "environment": environment_digest(),
        },
    )
    return ModelBundle(model=model, tokenizer=tokenizer, device=device, identity=identity)


def layer_checkpointing_is_consulted(model) -> bool | None:
    """Does any module of this model actually READ its checkpointing flag?

    ``PreTrainedModel.gradient_checkpointing_enable()`` sets
    ``module.gradient_checkpointing = True`` (and installs
    ``_gradient_checkpointing_func``) on every submodule that declares support.
    Whether that has any EFFECT depends on the modelling code: a decoder loop
    that never consults the flag makes the call a no-op.

    This inspects the source of every concrete module class that carries the
    flag and reports whether the flag (or the checkpointing function) is read
    anywhere outside its own assignment.  Returns ``None`` when the source is
    unavailable, which is recorded as UNKNOWN rather than assumed either way.
    """
    import inspect
    import re

    assignment = re.compile(r"self\.gradient_checkpointing\s*=(?!=)")
    seen_flag = False
    unknown = False
    for module in model.modules():
        if not hasattr(module, "gradient_checkpointing"):
            continue
        seen_flag = True
        try:
            source = inspect.getsource(type(module))
        except (OSError, TypeError):  # pragma: no cover - source-less class
            unknown = True
            continue
        # strip the flag's own assignments; anything left is a READ
        remainder = assignment.sub("", source)
        if ("self.gradient_checkpointing" in remainder
                or "_gradient_checkpointing_func" in remainder):
            return True
    if not seen_flag or unknown:
        return None
    return False


def enable_training_memory_savings(model) -> dict[str, Any]:
    """Gradient checkpointing + input grads (contract §22, training path).

    Two independent mechanisms could apply to this backbone:

    * **loop-level** checkpointing inside ``OuroModel.forward``, driven by
      ``config.rltt_loop_level_checkpointing`` and active while training with
      ``use_cache=False``.  This one is REAL on the frozen checkpoint: the
      forward wraps ``_run_single_ut_loop`` in ``torch.utils.checkpoint``;
    * **layer-level** checkpointing through the standard transformers API.
      ``OuroModel`` declares ``supports_gradient_checkpointing = True`` and
      initialises ``self.gradient_checkpointing = False``, but its decoder path
      never consults the flag, so ``gradient_checkpointing_enable()`` is a
      NO-OP on this architecture.

    The record therefore separates what was REQUESTED from what is ACTIVE, and
    the active value is DETECTED (:func:`layer_checkpointing_is_consulted`)
    rather than inferred from the API call succeeding.  Recording a no-op as an
    active memory-saving path would have put a false statement into every arm
    result and into the campaign manifest (review finding, minor (d)).
    """
    consulted = layer_checkpointing_is_consulted(model)
    state: dict[str, Any] = {
        "loop_level_checkpointing": bool(
            getattr(model.config, "rltt_loop_level_checkpointing", False)
        ),
        "layer_level_checkpointing_requested": False,
        "layer_level_checkpointing_active": False,
        "layer_level_checkpointing_detection": (
            "UNKNOWN" if consulted is None else
            ("CONSULTED_BY_FORWARD" if consulted else "NOT_CONSULTED_BY_FORWARD")
        ),
        "input_require_grads": False,
    }
    if getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        state["layer_level_checkpointing_requested"] = True
        state["layer_level_checkpointing_active"] = bool(consulted)
    #: kept for consumers that read the old key; it now carries the ACTIVE
    #: value, never the merely requested one.
    state["layer_level_checkpointing"] = state["layer_level_checkpointing_active"]
    if not state["layer_level_checkpointing_active"]:
        state["layer_level_checkpointing_note"] = (
            "gradient_checkpointing_enable() was called but this architecture's "
            "forward never consults the flag; only the loop-level RLTT "
            "checkpointing above is a real memory saving"
        )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        state["input_require_grads"] = True
    return state


def set_deterministic_eval() -> None:
    """Determinism knobs for evaluation paths (contract §22)."""
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
