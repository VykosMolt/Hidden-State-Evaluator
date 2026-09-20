#!/usr/bin/env bash
set -euo pipefail

cd /home/moloch/ouro_project
export HF_HOME=/home/moloch/ouro_project/shared/hf_cache
export HF_DATASETS_CACHE=/home/moloch/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1

out=opi/verification/paper_verification/powered_clean_probe_and_corecontent_20260711

shared/venv/bin/python shared/tools/paper_verification/powered_hh_pair_disjoint_probe.py extract --backbone base --pairs 40000 --output-dir "$out"
shared/venv/bin/python shared/tools/paper_verification/powered_hh_pair_disjoint_probe.py fit --backbone base --pairs 40000 --output-dir "$out"
shared/venv/bin/python shared/tools/paper_verification/powered_hh_pair_disjoint_probe.py extract --backbone thinking --pairs 40000 --output-dir "$out"
shared/venv/bin/python shared/tools/paper_verification/powered_hh_pair_disjoint_probe.py fit --backbone thinking --pairs 40000 --output-dir "$out"
shared/venv/bin/python shared/tools/paper_verification/powered_hh_pair_disjoint_probe.py extract --backbone rltt --pairs 40000 --output-dir "$out"
shared/venv/bin/python shared/tools/paper_verification/powered_hh_pair_disjoint_probe.py fit --backbone rltt --pairs 40000 --output-dir "$out"
