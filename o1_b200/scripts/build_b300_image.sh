#!/usr/bin/env bash
# Build the production B300 image locally (linux/amd64), record every
# environment fact, and emit CONTAINER_IMAGE_RECORD.json with the immutable
# image digest. Never embeds credentials; never includes the checkpoint.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WHEELS_SRC="${O1_B300_WHEELS:-/home/moloch/b200_build_cache/wheels_b300}"
IMAGE_NAME="${1:-o1-b300-runner}"
# Default from o1_b200/VERSION, never a frozen literal: a hardcoded
# default silently rebuilt and re-tagged an ALREADY PUSHED version and
# overwrote its image record with a different local id.
VERSION_TAG="${2:-v$(cat "$(dirname "${BASH_SOURCE[0]}")/../VERSION" | tr -d "[:space:]")}"
OUT="$ROOT/o1_b200/provider/runpod/CONTAINER_IMAGE_RECORD.json"

cd "$ROOT"
rm -rf build_ctx && mkdir -p build_ctx/wheels

# Foundation Learner source lives in a separate worktree, outside this build
# context, so it is staged in.  Refused rather than skipped by default: an
# image without it acquires an accelerator and then refuses at the FL
# handover, which is the expensive way to discover a packaging gap.  Set
# O1_B300_WITHOUT_FL=1 to deliberately build an O1-only image.
FL_SRC="${O1_B300_FL_SOURCE:-/home/moloch/ouro_worktrees/foundation-learner-b200-v0/foundation_learner}"
if [[ "${O1_B300_WITHOUT_FL:-0}" == "1" ]]; then
  FL_TREE_SHA="ABSENT_BY_REQUEST (O1_B300_WITHOUT_FL=1; combined sessions unsupported by this image)"
  mkdir -p build_ctx/foundation_learner
else
  if [[ ! -d "$FL_SRC" ]]; then
    echo "REFUSED: FL source $FL_SRC is absent; a combined O1 -> FL session" >&2
    echo "         cannot run from the resulting image.  Set" >&2
    echo "         O1_B300_FL_SOURCE, or O1_B300_WITHOUT_FL=1 to build an" >&2
    echo "         O1-only image deliberately." >&2
    exit 2
  fi
  # reports/ is local run output (~0.5 GB) and never belongs in the image;
  # the pregen corpus is fetched and hash-verified on the pod instead.
  tar -C "$(dirname "$FL_SRC")" -cf - \
      --exclude="reports" --exclude="__pycache__" --exclude="*.pyc" \
      "$(basename "$FL_SRC")" | tar -C build_ctx -xf -
  if [[ ! -x build_ctx/foundation_learner/deploy/fl_b200_entry.sh ]]; then
    echo "REFUSED: staged FL source has no executable deploy/fl_b200_entry.sh" >&2
    exit 2
  fi
  # LC_ALL=C: sort order is locale-dependent, so without it the SAME tree
  # digests differently on a differently-configured machine and the recorded
  # provenance hash stops being reproducible.  -print0/-0 handles newlines in
  # names; symlinks are listed explicitly so a link change is not invisible.
  # deploy/environment_lock.json RECORDS this digest (and INTEGRATION.md
  # quotes the image id), so they cannot be part of what is digested: a
  # digest that covered its own record could never be re-synced without
  # changing itself.  SHA256SUMS is excluded for the SAME reason, one step
  # removed: it covers environment_lock.json, so a lock sync changes the
  # sums, which would change this digest, which changes the image id the
  # lock records -- a cycle that never converges.  Excluding it loses no
  # coverage, because SHA256SUMS is derived: every file it lists is already
  # hashed individually into this digest.  Both files still ship in the image; they are only
  # excluded from the identity computation.
  FL_TREE_SHA=$(cd build_ctx/foundation_learner \
    && find . \( -type f -o -type l \) \
         ! -path ./deploy/environment_lock.json \
         ! -path ./deploy/INTEGRATION.md \
         ! -path ./SHA256SUMS -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum | LC_ALL=C sha256sum \
    | cut -d' ' -f1)
fi

# verify the frozen wheel set against its hash manifest BEFORE building
( cd "$WHEELS_SRC" && sha256sum --check --quiet WHEELS_B300.sha256 )
cp "$WHEELS_SRC"/*.whl build_ctx/wheels/
WHEELSET_SHA=$(sha256sum "$WHEELS_SRC/WHEELS_B300.sha256" | cut -d' ' -f1)
TORCH_WHEEL_SHA=$(grep -E " torch-" "$WHEELS_SRC/WHEELS_B300.sha256" | cut -d' ' -f1)
TRITON_WHEEL_SHA=$(grep -E " triton-" "$WHEELS_SRC/WHEELS_B300.sha256" | cut -d' ' -f1)

# resolve the base image to an immutable digest BEFORE building
docker pull --platform linux/amd64 python:3.14-slim-bookworm >/dev/null
BASE_DIGEST=$(docker inspect --format '{{index .RepoDigests 0}}' python:3.14-slim-bookworm)

# Identity of the executable o1_b200 source going into the image.  Computed by
# the SAME function pre_rental_check uses to verify it, so the two cannot
# drift.  Without this the build recorded a FL source hash and nothing at all
# for the driver code that spends the money.
O1_SRC_SHA=$(PYTHONPATH="$ROOT" python3 -c "from o1_b200.provider.runpod.pre_rental_check import _o1_source_tree_sha256 as h; print(h('$ROOT'))")
echo "o1_b200 source sha256: $O1_SRC_SHA"

docker build --platform linux/amd64 \
  --build-arg BASE_IMAGE="$BASE_DIGEST" \
  -f o1_b200/deploy/Dockerfile.b300 \
  -t "$IMAGE_NAME:$VERSION_TAG" .

PREV_IMAGE_ID="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('image_id_local',''))" "$ROOT/o1_b200/provider/runpod/CONTAINER_IMAGE_RECORD.json" 2>/dev/null || true)"
IMAGE_ID=$(docker inspect --format '{{.Id}}' "$IMAGE_NAME:$VERSION_TAG")
DOCKERFILE_SHA=$(sha256sum o1_b200/deploy/Dockerfile.b300 | cut -d' ' -f1)
LOCK_SHA=$(sha256sum o1_b200/deploy/requirements.b300.lock | cut -d' ' -f1)

# wheel-level native-arch verification on the CUDA host (containers build
# GPU-less): the arch list must carry the Blackwell targets and NO PTX
# (compute_*) entries — with no PTX, silent JIT fallback is impossible.
WHEELCHECK_VENV="${O1_B300_WHEELCHECK:-/home/moloch/b200_build_cache/wheelcheck_b300_venv}"
ARCH_LIST=$("$WHEELCHECK_VENV/bin/python" -c "import torch, json; al = torch.cuda.get_arch_list(); assert 'sm_100' in al and 'sm_120' in al, al; assert not any(a.startswith('compute_') for a in al), ('PTX present', al); print(json.dumps(al))" 2>/dev/null)
echo "wheel arch list (host verification): $ARCH_LIST"

VERSIONS=$(docker run --rm --entrypoint /opt/venv/bin/python "$IMAGE_NAME:$VERSION_TAG" -c "
import json, platform, sys, torch, transformers, numpy
print(json.dumps({
  'python': sys.version.split()[0],
  'torch': torch.__version__,
  'torch_cuda_runtime': torch.version.cuda,
  'cudnn': torch.backends.cudnn.version(),
  'transformers': transformers.__version__,
  'numpy': numpy.__version__,
  'arch_list_source': 'host wheel verification (build has no GPU); re-verified on the acquired GPU by deploy/hardware_gate.py',
  'arch_list': json.loads('$ARCH_LIST'),
  'sm103_native_basis': 'sm_100 cubins run natively on sm_103 (NVIDIA same-major/higher-minor SASS compatibility rule) and the wheel additionally carries 59 sm_103a arch-tuned cubins; NO PTX embedded so JIT fallback is impossible. arch_list above is torch.cuda.get_arch_list(), i.e. the build TARGET list, which does not enumerate arch-specific a-suffixed variants: the sm_103a/sm_100a cubin counts are cuobjdump evidence in deploy/FATBINARY_ARCH_EVIDENCE.json, and the loaded kernels are exercised on the acquired GPU by deploy/hardware_gate.py',
  'glibc': platform.libc_ver()[1],
}))")

python3 - "$OUT" <<EOF
import json, sys, time
out = sys.argv[1]
record = {
  "schema": "o1b300.container_image_record.v2",
  "build_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "platform": "linux/amd64",
  "image_name": "$IMAGE_NAME:$VERSION_TAG",
  "image_id_local": "$IMAGE_ID",
  "registry_digest_ref": "UNRESOLVED_UNTIL_PUSH (docker push prints repo@sha256:...)",
  "base_image_digest": "$BASE_DIGEST",
  "target_profiles": {"B300": "sm_103 / CC 10.3 / 288 GB (primary)",
                      "B200": "sm_100 / CC 10.0 / 180 GB (explicit fallback)"},
  "supported_driver_range": ">= r580 (CUDA 13.0 requirement; RunPod B300/B200 hosts report CUDA 13.0/13.2)",
  "dockerfile_sha256": "$DOCKERFILE_SHA",
  "dependency_lock_sha256": "$LOCK_SHA",
  "wheelset_manifest_sha256": "$WHEELSET_SHA",
  "o1_b200_source_sha256": "$O1_SRC_SHA",
  "foundation_learner_source_sha256": "$FL_TREE_SHA",
  "foundation_learner_layout": "/opt/foundation_learner/foundation_learner (import root /opt/foundation_learner); pregen episode corpus NOT baked — campaign/fetch_pregen.py materialises and hash-verifies it on the pod",
  "torch_wheel_sha256": "$TORCH_WHEEL_SHA",
  "triton_wheel_sha256": "$TRITON_WHEEL_SHA",
  "environment": json.loads('''$VERSIONS'''),
  "system_packages": "base python:3.14-slim-bookworm only; no extra apt packages installed",
  "notes": "no FP8/FP4/quantization/speculative/vLLM/TensorRT-LLM/SGLang; no startup installation/compilation/clone/resolution; checkpoint never baked; no credentials",
}
with open(out, "w") as fh:
    json.dump(record, fh, indent=2, sort_keys=True); fh.write("\n")
print(json.dumps(record["environment"], indent=1))
print("image id:", record["image_id_local"])
EOF
rm -rf build_ctx
# The four operator documents quote the local image id; rewrite them from
# the record so a documented cross-check can never name an image that no
# longer exists (test_launch_path pins this correspondence).
# Anchored to the PREVIOUS local id (read from the record before it was
# overwritten) so a registry digest or any other sha256 in those documents
# is never rewritten.
for doc in "$ROOT/o1_b200/README.md" "$ROOT/o1_b200/deploy/README.md" \
           "$ROOT/o1_b200/provider/runpod/PRE_RENTAL_PREREQUISITES.md" \
           "$ROOT/o1_b200/provider/runpod/REGISTRY_PUSH_PROCEDURE.md"; do
  if [[ -n "${PREV_IMAGE_ID:-}" ]]; then
    sed -i "s|$PREV_IMAGE_ID|$IMAGE_ID|g" "$doc"
  fi
done
echo "BUILD OK -> $OUT"
