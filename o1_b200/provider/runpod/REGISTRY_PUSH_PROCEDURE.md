# Registry Push — the one remaining non-GPU staging action

The B300 production image is built locally as `o1-b300-runner:v0.3.7`
(identity-recorded in `CONTAINER_IMAGE_RECORD.json`, local image id
`sha256:55f2c45ddc6cbfe20771c7cfc20cdea503724a40fe82bf3e0265776b6c836f2d`,
built by `scripts/build_b300_image.sh` from `Dockerfile.b300`). The registry
digest is UNRESOLVED until the operator pushes; it was **not** pushed in the
pre-rental task because:

- a fresh GHCR container package defaults to **private**; a multi-GB private
  package exceeds the 500 MB private free tier, creating a blocked-or-billable
  storage state (forbidden: no billable resource in this task);
- GHCR package visibility cannot be reliably switched to public via the API
  before the first push.

## Exact procedure (run once, before rental, ~10 minutes)

1. In the GitHub UI, pre-create the package as **public** by pushing a tiny
   placeholder OR simply push and immediately set
   `https://github.com/users/VykosMolt/packages/container/o1-b300-runner/settings`
   → *Change visibility* → **Public** (public GHCR storage is free).
2. Push:

```sh
gh auth token | docker login ghcr.io -u VykosMolt --password-stdin
docker tag o1-b300-runner:v0.3.7 ghcr.io/vykosmolt/o1-b300-runner:v0.3.7
docker push ghcr.io/vykosmolt/o1-b300-runner:v0.3.7
```

3. Record the **registry digest** printed by the push (`…@sha256:…`) into
   `RUNPOD_SESSION_CONFIG.json` (`image_digest_ref`) and regenerate the
   canonical pod-request render. The image contains no secrets and no model
   checkpoint; public visibility is consistent with the (already public)
   repository content. Mutable-tag references remain refused by the adapter —
   only the `@sha256:` digest form is accepted.
