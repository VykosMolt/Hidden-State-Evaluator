#!/usr/bin/env bash
# Build the production B300 image locally (linux/amd64), record every
# environment fact, and emit CONTAINER_IMAGE_RECORD.json with the immutable
# image digest. Never embeds credentials; never includes the checkpoint.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WHEELS_SRC="${O1_B300_WHEELS:-/home/moloch/b200_build_cache/wheels_b300}"
IMAGE_NAME="${1:-o1-b300-runner}"
VERSION_TAG="${2:-v0.3.0}"
OUT="$ROOT/o1_b200/provider/runpod/CONTAINER_IMAGE_RECORD.json"

cd "$ROOT"
rm -rf build_ctx && mkdir -p build_ctx/wheels

# verify the frozen wheel set against its hash manifest BEFORE building
( cd "$WHEELS_SRC" && sha256sum --check --quiet WHEELS_B300.sha256 )
cp "$WHEELS_SRC"/*.whl build_ctx/wheels/
WHEELSET_SHA=$(sha256sum "$WHEELS_SRC/WHEELS_B300.sha256" | cut -d' ' -f1)
TORCH_WHEEL_SHA=$(grep -E " torch-" "$WHEELS_SRC/WHEELS_B300.sha256" | cut -d' ' -f1)
TRITON_WHEEL_SHA=$(grep -E " triton-" "$WHEELS_SRC/WHEELS_B300.sha256" | cut -d' ' -f1)

# resolve the base image to an immutable digest BEFORE building
docker pull --platform linux/amd64 python:3.14-slim-bookworm >/dev/null
BASE_DIGEST=$(docker inspect --format '{{index .RepoDigests 0}}' python:3.14-slim-bookworm)

docker build --platform linux/amd64 \
  --build-arg BASE_IMAGE="$BASE_DIGEST" \
  -f o1_b200/deploy/Dockerfile.b300 \
  -t "$IMAGE_NAME:$VERSION_TAG" .

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
  'sm103_native_basis': 'sm_100 cubins run natively on sm_103 (NVIDIA same-major/higher-minor SASS rule); wheel additionally carries sm_103a arch-tuned kernels; NO PTX embedded so JIT fallback is impossible',
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
echo "BUILD OK -> $OUT"
