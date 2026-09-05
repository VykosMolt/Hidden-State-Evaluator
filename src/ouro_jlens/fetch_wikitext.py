"""Fetch an immutable, revision-pinned WikiText fitting corpus.

The old helper delegated to ``jlens.examples`` without pinning the dataset
revision and overwrote the canonical prompt file directly.  This command
records the exact Hub revision and refuses both partial resume and replacement.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from ouro_jlens.evidence import atomic_write_json, file_record, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DEFAULT_OUT = (
    PROJECT_ROOT / "artifacts" / "jlens" / "data" / "wikitext_prompts_b08601e.json"
)


def _record(path: Path, identity: str) -> dict:
    result = file_record(path)
    result["path"] = identity
    return result


def _validate_existing(out: Path, provenance: Path, *, n: int, min_chars: int, revision: str) -> None:
    if out.is_symlink() or provenance.is_symlink() or not out.is_file() or not provenance.is_file():
        raise ValueError("prompt corpus resume is partial, missing, or linked")
    try:
        document = json.loads(provenance.read_text(encoding="utf-8"))
        prompts = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("prompt corpus or provenance is unreadable") from exc
    expected = {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "revision": revision,
        "minimum_characters": min_chars,
        "requested_prompts": n,
    }
    if document.get("schema_version") != 1 \
            or document.get("status") != "FRESH_PINNED_DATASET_REVISION" \
            or document.get("source") != expected:
        raise ValueError("prompt corpus provenance does not match the requested frozen source")
    try:
        datasets_version = importlib.metadata.version("datasets")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("datasets runtime is unavailable for prompt-corpus verification") from exc
    if document.get("datasets_version") != datasets_version:
        raise ValueError("prompt corpus generator runtime version changed")
    source_path = Path(__file__).resolve()
    generator = document.get("generator")
    expected_generator = _record(source_path, "src/ouro_jlens/fetch_wikitext.py")
    if generator != expected_generator:
        raise ValueError("prompt corpus generator byte identity mismatch")
    record = document.get("output")
    if not isinstance(record, dict) or record.get("path") != "wikitext_prompts" \
            or record.get("size") != out.stat().st_size or record.get("sha256") != sha256_file(out):
        raise ValueError("prompt corpus byte identity mismatch")
    if not isinstance(prompts, list) or len(prompts) != n \
            or any(not isinstance(value, str) or len(value.strip()) < min_chars for value in prompts):
        raise ValueError("prompt corpus contents do not match the frozen selection rule")


def fetch(n: int, min_chars: int, revision: str) -> list[str]:
    if n <= 0 or min_chars <= 0:
        raise ValueError("n and min-chars must be positive")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("dataset revision must be a lowercase 40-character commit hash")
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        revision=revision,
        streaming=True,
    )
    prompts: list[str] = []
    for record in dataset:
        text = record.get("text")
        if isinstance(text, str) and len(text.strip()) >= min_chars:
            prompts.append(text)
            if len(prompts) == n:
                break
    if len(prompts) != n:
        raise RuntimeError(f"pinned dataset yielded only {len(prompts)} of {n} requested prompts")
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", nargs="?", type=int, default=1200)
    parser.add_argument("--min-chars", type=int, default=600)
    parser.add_argument("--revision", default=WIKITEXT_REVISION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out
    provenance = out.with_suffix(".provenance.json")
    if out.exists() or provenance.exists() or out.is_symlink() or provenance.is_symlink():
        _validate_existing(
            out, provenance, n=args.n, min_chars=args.min_chars, revision=args.revision
        )
        print(f"verified existing {args.n}-prompt corpus at {out}")
        return 0

    prompts = fetch(args.n, args.min_chars, args.revision)
    atomic_write_json(out, prompts, indent=None)
    source_path = Path(__file__).resolve()
    atomic_write_json(provenance, {
        "schema_version": 1,
        "status": "FRESH_PINNED_DATASET_REVISION",
        "source": {
            "dataset": "Salesforce/wikitext",
            "config": "wikitext-103-raw-v1",
            "split": "train",
            "revision": args.revision,
            "minimum_characters": args.min_chars,
            "requested_prompts": args.n,
        },
        "datasets_version": importlib.metadata.version("datasets"),
        "generator": _record(source_path, "src/ouro_jlens/fetch_wikitext.py"),
        "output": _record(out, "wikitext_prompts"),
    })
    _validate_existing(out, provenance, n=args.n, min_chars=args.min_chars, revision=args.revision)
    print(f"saved {len(prompts)} pinned prompts to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
