#!/usr/bin/env bash
# Reproducible environment setup on a fresh GPU host.
# The caller must have verified MANIFEST.sha256 and PINNED_INPUTS.json first.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Refuse to execute an archive whose source/input identity was not verified.
"$PY" -m ouro_jlens.publish stage-verify --root . --allow-extra

# Keep every runtime dependency version explicit.  This is intentionally not
# a floating "compatible" install: a result without these pins is not the
# same experiment.
"$PY" -m pip install -q \
  "transformers==4.54.1" \
  "numpy==2.4.3" \
  "scikit-learn==1.8.0" \
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
"$PY" -m pip install -q -e "$JLENS_DIR" --no-deps
"$PY" -m ouro_jlens.publish runtime-verify

MODEL_REPOSITORY=ByteDance/Ouro-2.6B
MODEL_REVISION=1ed04250da1a9936042725d302e81c8fa2ab5abd
hf download "$MODEL_REPOSITORY" --revision "$MODEL_REVISION" \
  --cache-dir artifacts/hf_cache/hub --quiet
test -f "artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/snapshots/$MODEL_REVISION/model.safetensors"

# Print only non-secret environment facts after all pins and GPU checks pass.
"$PY" -c 'import importlib.metadata as md, sys, torch, transformers, jlens; expected = {"transformers": "4.54.1", "numpy": "2.4.3", "scikit-learn": "1.8.0", "matplotlib": "3.10.8", "safetensors": "0.7.0", "accelerate": "1.13.0", "huggingface-hub": "0.36.2"}; found = {name: md.version(name) for name in expected}; assert found == expected, ("dependency pin mismatch", found, expected); assert sys.version_info[:2] == (3, 11); assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__, torch.cuda.get_device_name())'
