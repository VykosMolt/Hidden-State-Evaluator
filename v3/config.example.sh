# Copy to config.sh, edit the paths, then `source config.sh` before running any script.
# Every required input is listed with its SHA-256 and size in INPUTS_OFFLINE.sha256.

# rpe_v3_runs/: evaluator per-pair shards (A_thinking_1024/, A_rltt/) and probe features (thinking_left/, rltt_left_cloud/)
export RPE_RUNS=/path/to/rpe_v3_runs
# 40,000-pair selection and pair-disjoint split, plus version 2's probe results
# (hh_train_indices_40000.json, hh_{base,thinking,rltt}_40000_pair_disjoint_{probe.pt,results.json}, ...)
export RPE_SPLIT=/path/to/powered_clean_probe_and_corecontent_20260711
# version 2's evaluator audit on the full test split (full_hh_antisymmetry.json, full_hh_antisymmetry_rows.csv)
export RPE_AUDIT=/path/to/full_hh_audit
# version 1's evaluator checkpoints pairwise_epoch1.pt ... pairwise_epoch5.pt (GPU steps only)
export RPE_CKPT_DIR=/path/to/checkpoints_pairwise
# local Ouro-RLTT checkpoint directory: weights (GPU steps) and the tokenizer used for every backbone
export RPE_RLTT_MODEL=/path/to/ouro_rltt_local
# where results are written (default: ./results in this package)
# export RPE_RESULTS=/path/to/results

# HF cache for HH-RLHF (revision pinned in scripts/config.py) and Ouro-2.6B-Thinking (GPU steps)
export HF_HOME=/path/to/hf_cache
export OMP_NUM_THREADS=8
