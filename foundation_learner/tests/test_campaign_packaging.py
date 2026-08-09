"""Packaging and manifest filling (contract §19).

All checks run against a synthetic repository tree in a temporary directory, so
the packaging rules are exercised without building the 500 MB release.
"""
from __future__ import annotations

import json
import os
import zipfile

import pytest

from foundation_learner.scripts import make_manifest as mm
from foundation_learner.scripts import package_release as pr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
PACKAGE = os.path.join(REPO, "foundation_learner")


def fake_repo(tmp_path):
    package = tmp_path / "foundation_learner"
    (package / "campaign").mkdir(parents=True)
    (package / "reports" / "local_runs").mkdir(parents=True)
    (package / "__pycache__").mkdir()
    (package / "campaign" / "scheduler.py").write_text("x = 1\n", encoding="utf-8")
    (package / "VERSION").write_text("0.1.0\n", encoding="utf-8")
    (package / "reports" / "local_runs" / "junk.json").write_text("{}",
                                                                  encoding="utf-8")
    (package / "__pycache__" / "a.pyc").write_text("junk", encoding="utf-8")
    (package / "campaign" / "stale.pyc").write_text("junk", encoding="utf-8")
    pregen = tmp_path / "artifacts_fl" / "pregen"
    (pregen / "episodes" / "TRAIN").mkdir(parents=True)
    (pregen / "episodes" / "TRAIN" / "f.jsonl").write_text('{"e": 1}\n',
                                                           encoding="utf-8")
    for name in pr.PREGEN_MANIFEST_FILES:
        (pregen / name).write_text(json.dumps({"name": name}) + "\n",
                                   encoding="utf-8")
    return str(tmp_path), str(pregen)


def test_the_content_manifest_excludes_scratch_and_bytecode(tmp_path):
    repo, pregen = fake_repo(tmp_path)
    pr.mirror_pregen_manifests(pregen)
    rels = [os.path.relpath(p, repo).replace(os.sep, "/")
            for p in pr.content_manifest(repo, pregen_root=pregen)]
    assert "foundation_learner/campaign/scheduler.py" in rels
    assert "artifacts_fl/pregen/episodes/TRAIN/f.jsonl" in rels
    assert not any("local_runs" in r for r in rels)
    assert not any("__pycache__" in r for r in rels)
    assert not any(r.endswith(".pyc") for r in rels)


def test_pregen_manifests_are_mirrored_and_hash_verified(tmp_path):
    repo, pregen = fake_repo(tmp_path)
    mirrored = pr.mirror_pregen_manifests(pregen)
    assert set(mirrored) == set(pr.PREGEN_MANIFEST_FILES)
    for name in pr.PREGEN_MANIFEST_FILES:
        target = os.path.join(pregen, pr.MANIFEST_DIR_NAME, name)
        assert os.path.isfile(target)
    # a mirror that has gone stale is refreshed, never silently accepted
    with open(os.path.join(pregen, "SHARD_SUMS.json"), "w",
              encoding="utf-8") as fh:
        fh.write('{"changed": true}\n')
    again = pr.mirror_pregen_manifests(pregen)
    assert again["SHARD_SUMS.json"] != mirrored["SHARD_SUMS.json"]


def test_incomplete_pregeneration_is_refused(tmp_path):
    repo, pregen = fake_repo(tmp_path)
    os.remove(os.path.join(pregen, "SHARD_SUMS.json"))
    with pytest.raises(pr.PackagingError) as exc:
        pr.mirror_pregen_manifests(pregen)
    assert "incomplete" in str(exc.value)


def test_a_missing_pregen_root_refuses_a_release(tmp_path):
    repo, _ = fake_repo(tmp_path)
    with pytest.raises(pr.PackagingError):
        pr.content_manifest(repo, pregen_root=str(tmp_path / "nope"))
    code_only = pr.content_manifest(repo, pregen_root=None,
                                    include_pregen=False)
    assert code_only


def test_build_is_deterministic_and_exactly_covered(tmp_path):
    repo, pregen = fake_repo(tmp_path)
    out = os.path.join(repo, "dist")
    first = pr.build(repo, pregen_root=pregen, out_dir=out)
    second = pr.build(repo, pregen_root=pregen, out_dir=out)
    assert first["zip_sha256"] == second["zip_sha256"]
    assert first["coverage"]["ok"] is True
    with zipfile.ZipFile(first["zip_path"]) as zf:
        names = zf.namelist()
        assert names == sorted(names)
        assert "foundation_learner/SHA256SUMS" in names
        for info in zf.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.external_attr == 0o644 << 16
    assert open(first["zip_path"] + ".sha256", encoding="utf-8").read() == \
        f"{first['zip_sha256']}  {pr.ZIP_NAME}\n"


def test_exact_coverage_detects_an_added_or_removed_file(tmp_path):
    repo, pregen = fake_repo(tmp_path)
    out = os.path.join(repo, "dist")
    report = pr.build(repo, pregen_root=pregen, out_dir=out)
    sums = report["sha256sums_path"]
    extra = os.path.join(repo, "foundation_learner", "campaign", "new.py")
    open(extra, "w").write("y = 2\n")
    with pytest.raises(pr.PackagingError) as exc:
        pr.verify_exact_coverage(sums, repo, pregen_root=pregen)
    assert "present-but-unlisted" in str(exc.value)
    os.remove(extra)
    os.remove(os.path.join(repo, "foundation_learner", "VERSION"))
    with pytest.raises(pr.PackagingError) as exc:
        pr.verify_exact_coverage(sums, repo, pregen_root=pregen)
    assert "listed-but-absent" in str(exc.value)


def test_exact_coverage_detects_a_modified_file(tmp_path):
    repo, pregen = fake_repo(tmp_path)
    report = pr.build(repo, pregen_root=pregen,
                      out_dir=os.path.join(repo, "dist"))
    with open(os.path.join(repo, "foundation_learner", "VERSION"), "w") as fh:
        fh.write("9.9.9\n")
    with pytest.raises(pr.PackagingError) as exc:
        pr.verify_exact_coverage(report["sha256sums_path"], repo,
                                 pregen_root=pregen)
    assert "hash mismatches" in str(exc.value)


# ---------------- manifest ----------------

def test_the_shipped_template_only_leaves_b200_fields_open():
    with open(os.path.join(PACKAGE, mm.TEMPLATE_NAME), encoding="utf-8") as fh:
        template = json.load(fh)
    open_fields = mm.unresolved_fields(template)
    non_b200 = [f for f in open_fields
                if not f.startswith("b200_derived_unresolved")]
    assert sorted(non_b200) == [
        "ecology.generator_source_hashes",
        "ecology.shard_sums_sha256",
        "ecology.split_manifest_sha256",
        "package.source_commit",
        "package.zip_sha256",
    ], non_b200


def test_filling_the_manifest_resolves_exactly_the_pre_session_fields(tmp_path):
    pregen = os.path.join(REPO, "artifacts_fl", "pregen_tiny")
    if not os.path.isfile(os.path.join(pregen, "PREGEN_MANIFEST.json")):
        pytest.skip("no tiny pregeneration present")
    with open(os.path.join(PACKAGE, mm.TEMPLATE_NAME), encoding="utf-8") as fh:
        template = json.load(fh)
    zip_path = tmp_path / "FOUNDATION_LEARNER_B200_V0.1.0.zip"
    zip_path.write_bytes(b"not a real zip, but a real digest\n")
    manifest = mm.fill(template, pregen_root=pregen, repo_root=REPO,
                       zip_path=str(zip_path))
    status = mm.check_only_b200_fields_remain(manifest)
    assert status["non_b200_unresolved"] == [], status
    assert len(manifest["ecology"]["split_manifest_sha256"]) == 64
    assert len(manifest["ecology"]["generator_source_hashes"]) == 12
    assert len(manifest["package"]["source_commit"]) == 40
    assert manifest["b200_derived_unresolved"][
        "measured_throughput_tokens_per_second"] == "UNRESOLVED_B200_BENCH"
    assert manifest["b200_derived_unresolved"][
        "o1_entry_command"] == "UNRESOLVED_OPERATOR_BOUND"


def test_a_mismatched_split_manifest_refuses_the_fill(tmp_path):
    pregen = tmp_path / "pregen"
    (pregen / pr.MANIFEST_DIR_NAME).mkdir(parents=True)
    from foundation_learner.ecology.manifests import write_split_manifest

    write_split_manifest(str(pregen / "family_split_manifest.json"))
    (pregen / "SHARD_SUMS.json").write_text("{}\n", encoding="utf-8")
    (pregen / "PREGEN_MANIFEST.json").write_text(
        json.dumps({"split_manifest_sha256": "c" * 64, "files": []}) + "\n",
        encoding="utf-8")
    with open(os.path.join(PACKAGE, mm.TEMPLATE_NAME), encoding="utf-8") as fh:
        template = json.load(fh)
    with pytest.raises(mm.ManifestError) as exc:
        mm.fill(template, pregen_root=str(pregen), repo_root=REPO)
    assert "DIFFERENT family split manifest" in str(exc.value)
