#!/bin/bash
# Runs the full pipeline in order. Run from any directory; all paths are relative to this
# script's directory (override with MAGMOM_ROOT). Requires MP_API_KEY for step 1 only.
# Step 1 is skipped if data/mp_magmom_raw.json already exists; the raw MP/JARVIS/NEMAD
# dumps are not shipped (see README), so steps 1-2 need network access.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
[ -f data/mp_magmom_raw.json ] || $PY build_mp_magmom_table.py
for s in phase1_pipeline subgroup_variance_controlled hubbard_u_feature \
         domain_adversarial_nn dann_feature_attribution domain_flag_baseline \
         targeted_finetuning finetuning_extended_scale finetuning_more_seeds \
         multimodel_panel multimodel_panel_extended nemad_ordering_validation \
         mp_jarvis_nemad_direct_comparison selective_prediction \
         make_figures make_graphical_abstract; do
  echo "== $s"
  $PY $s.py
done
