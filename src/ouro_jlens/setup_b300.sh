#!/usr/bin/env bash
# Reproducible environment setup on a fresh GPU host.
# The caller must have verified MANIFEST.sha256 and PINNED_INPUTS.json first.
set -euo pipefail

# Setup installs packages, checks git, and downloads a public model.  It must
# never inherit a credential, even when invoked directly outside pod_entry.
unset HF_TOKEN JLENS_HF_TOKEN_BOOTSTRAP

cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# The controller passes the digest it approved.  Refuse an unset, tag-based,
# or otherwise different runtime before touching the stage or model cache.
JLENS_IMAGE_DIGEST=${JLENS_IMAGE_DIGEST:?JLENS_IMAGE_DIGEST is required}
export JLENS_IMAGE_DIGEST
EXPECTED_IMAGE='runpod/pytorch@sha256:bbe1496e2215cca3d25a5e5cd291d31ea86603e4577a81eb40096787a50e5303'
if [[ "$JLENS_IMAGE_DIGEST" != "$EXPECTED_IMAGE" ]]; then
  echo "JLENS_IMAGE_DIGEST does not match the approved image" >&2
  exit 2
fi
"$PY" -c '
from ouro_jlens.publish import RUNTIME_IMAGE
import sys
if sys.argv[1] != RUNTIME_IMAGE:
    raise SystemExit("JLENS_IMAGE_DIGEST does not match the approved image")
' "$JLENS_IMAGE_DIGEST"
test -f src/ouro_jlens/runtime.lock.json

# Refuse to execute an archive whose source/input identity was not verified.
"$PY" -m ouro_jlens.publish stage-verify --root . --allow-extra

# Keep every runtime dependency version explicit.  This is intentionally not
# a floating "compatible" install: a result without these pins is not the
# same experiment.
"$PY" -m pip install -q --break-system-packages \
  "transformers==4.54.1" \
  "numpy==2.4.3" \
  "scikit-learn==1.8.0" \
  "threadpoolctl==3.6.0" \
  "matplotlib==3.10.8" \
  "safetensors==0.7.0" \
  "accelerate==1.13.0" \
  "huggingface_hub==0.36.2"

JLENS_DIR="$HOME/jacobian-lens"
JLENS_REPOSITORY=https://github.com/anthropics/jacobian-lens.git
JLENS_REVISION=581d398613e5602a5af361e1c34d3a92ea82ba8e
if [[ ! -d "$JLENS_DIR/.git" ]]; then
  git clone --quiet "$JLENS_REPOSITORY" "$JLENS_DIR"
fi
git -C "$JLENS_DIR" fetch --quiet --depth=1 origin "$JLENS_REVISION"
git -C "$JLENS_DIR" checkout --quiet --detach "$JLENS_REVISION"
if [[ -n "$(git -C "$JLENS_DIR" status --porcelain --untracked-files=all)" ]]; then
  echo "jacobian-lens checkout is dirty; refusing unpinned code" >&2
  exit 2
fi
test "$(git -C "$JLENS_DIR" rev-parse HEAD)" = "$JLENS_REVISION"
# The pinned jlens revision declares transformers>=5.5, but the Ouro runtime
# is intentionally pinned to transformers==4.54.1 for model compatibility.
# Install jlens without dependency resolution, then execute the explicit
# metadata check so this incompatibility is visible and fails closed.
JLENS_TRANSFORMERS_REQUIREMENT='transformers>=5.5'
JLENS_TRANSFORMERS_RUNTIME='4.54.1'
echo "jlens compatibility override: ${JLENS_TRANSFORMERS_REQUIREMENT} -> transformers==${JLENS_TRANSFORMERS_RUNTIME}"
"$PY" -m pip install -q -e "$JLENS_DIR" --no-deps --break-system-packages
"$PY" -m ouro_jlens.publish runtime-verify

MODEL_REPOSITORY=ByteDance/Ouro-2.6B
MODEL_REVISION=1ed04250da1a9936042725d302e81c8fa2ab5abd
hf download "$MODEL_REPOSITORY" --revision "$MODEL_REVISION" \
  --cache-dir artifacts/hf_cache/hub --quiet
test -f "artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/snapshots/$MODEL_REVISION/model.safetensors"

# The future digest-pinned RunPod image supplies this exact Torch build.  Do
# not let a dependency install silently replace it: the CUDA build and Python
# package version are part of the model/runtime identity.
EXPECTED_TORCH_VERSION='2.8.0+cu128'
EXPECTED_TORCH_CUDA='12.8'
"$PY" -c '
import importlib.metadata as md
import sys
import torch, transformers, jlens

expected = {
    "transformers": "4.54.1",
    "numpy": "2.4.3",
    "scikit-learn": "1.8.0",
    "threadpoolctl": "3.6.0",
    "matplotlib": "3.10.8",
    "safetensors": "0.7.0",
    "accelerate": "1.13.0",
    "huggingface-hub": "0.36.2",
}
found = {name: md.version(name) for name in expected}
if found != expected:
    raise SystemExit(f"dependency pin mismatch: {found!r} != {expected!r}")
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python version mismatch: {sys.version_info[:2]!r} != (3, 12)")
if torch.__version__ != sys.argv[1]:
    raise SystemExit(
        f"Torch build does not match the digest-pinned image: {torch.__version__!r} != {sys.argv[1]!r}"
    )
if torch.version.cuda != sys.argv[2]:
    raise SystemExit(
        f"Torch CUDA build does not match the digest-pinned image: {torch.version.cuda!r} != {sys.argv[2]!r}"
    )
if not torch.backends.cuda.is_built():
    raise SystemExit("Torch was not built with CUDA support")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available at runtime")
print(torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.get_device_name())
' "$EXPECTED_TORCH_VERSION" "$EXPECTED_TORCH_CUDA"
