# Cross-database machine learning for magnetic moments (Materials Project vs. JARVIS-DFT)

Code and data for the magnetic-moment cross-database transfer study. Composition-only (Magpie) models are trained on magnetic moments per atom from the Materials Project (MP) and JARVIS-DFT and evaluated within and across databases. Transfer fails mainly for compounds containing correlated-electron transition-metal elements, consistent with the absence of a Hubbard-U correction in the JARVIS-DFT workflow. The scripts cover the baseline transfer test, the subgroup diagnosis, four tested remedies (with selective prediction, five responses in total), a multi-model robustness panel, validation against NEMAD experimental magnetic orderings, and selective prediction.

## Run order

`run_all.sh` runs every step below in order. All paths are relative to the repository directory (set `MAGMOM_ROOT` to use another working directory); figures are written to `figures/`. The Materials Project and JARVIS-DFT tables were retrieved on 28 October 2026 (`mp-api` 0.46.5, `jarvis-tools` 2026.6.12). In every analysis, entries sharing a reduced formula are collapsed to their median per-atom moment (101,021 MP entries to 74,461 formulas; 88,811 JARVIS entries to 62,763).

1. `build_mp_magmom_table.py`: queries MP for total magnetization (requires the `MP_API_KEY` environment variable).
2. `phase1_pipeline.py` : Magpie featurization of both databases, baseline 4-condition transfer test, subgroup diagnosis, composition-overlap and leakage check.
3. `subgroup_variance_controlled.py`: MAE/target-std comparison of correlated-electron and other compounds, with composition-grouped bootstrap confidence intervals.
4. `hubbard_u_feature.py`: remedy 1, explicit Hubbard-U sensitivity feature (U values from pymatgen's MPRelaxSet).
5. `domain_adversarial_nn.py`: remedy 2, domain-adversarial neural network with gradient reversal.
6. `dann_feature_attribution.py`: input-gradient attribution comparison of the domain-adversarial and plain networks.
7. `domain_flag_baseline.py`: remedy 3, multitask model with a database-identity flag.
8. `targeted_finetuning.py`: remedy 4, targeted vs. random addition of target-domain data (k = 10, 100).
9. `finetuning_extended_scale.py`: the same comparison at k = 500, 2000, 5000.
10. `finetuning_more_seeds.py`: seeds 3-9 for k = 2000 and 5000 (10 seeds in total).
11. `multimodel_panel.py` and `multimodel_panel_extended.py`: 5-model and 9-model panels (15 models in total with the baseline GBM).
12. `nemad_ordering_validation.py`: JARVIS moments vs. NEMAD FM/AFM/NM ordering labels.
13. `mp_jarvis_nemad_direct_comparison.py`: MP and JARVIS compared on the compounds present in all three databases.
14. `selective_prediction.py`: bootstrap-ensemble uncertainty and abstention, MP to JARVIS.
15. `make_figures.py`, `make_graphical_abstract.py`: figures.

## Data files (`data/`)

- `mp_magmom_clean.csv`, `jarvis_magmom_clean.csv`: cleaned per-atom magnetic-moment tables.
- `predictions_mp_to_jarvis.csv`, `predictions_jarvis_to_mp.csv`: per-compound true and predicted moments for the zero-shot transfers, with the `correlated_electron` flag.
- `baseline_4condition.json`, `hyperparam_sweep.json`, `subgroup_diagnosis.json`, `subgroup_variance_controlled.json`: baseline transfer and subgroup results.
- `overlap_formulas.json`, `overlap_leakage_check.json`: composition overlap between the databases and the leakage check.
- `hubbard_u_feature_results.json`: Hubbard-U feature results.
- `domain_adversarial_results.json`, `dann_feature_attribution.json`: domain-adversarial network results and attribution analysis.
- `multitask_domain_flag_baseline.json`: database-flag model results.
- `targeted_finetuning_results.json`, `targeted_finetuning_raw_checkpoint.json`, `finetuning_extended_scale.json`, `finetuning_extended_raw_checkpoint.json`, `finetuning_more_seeds_merged.json`, `finetuning_more_seeds_raw_checkpoint.json`: targeted vs. random data-addition results and raw checkpoints.
- `multimodel_panel_results.json`, `multimodel_panel_extended.json`, `multimodel_panel_combined_all15.json`: multi-model panel results.
- `nemad_overlap_check.json`, `jarvis_ordering_validation.json`, `mp_ordering_validation.json`, `mp_vs_jarvis_vs_nemad_direct.json`: NEMAD validation results.
- `hubbard_u_vs_magnetic_init_metadata.json`: comparison of MP and JARVIS calculation settings (Hubbard-U and magnetic initialization) from package source code.
- `selective_prediction_results.json`: selective-prediction results.

## Raw input files

`data/mp_magmom_raw.json` (Materials Project query result, fields `material_id`, `formula_pretty`, `total_magnetization`, `energy_above_hull`, `nsites`, `theoretical`, `is_stable`; Materials Project data are released under CC BY 4.0) and `data/jarvis_magmom_raw.json` (the `magmom_oszicar`, composition and atom-count fields of the JARVIS-DFT `dft_3d` table; JARVIS data are openly distributed by NIST) are included, both retrieved on 28 October 2026. They can be regenerated with `build_mp_magmom_table.py` (needs a free `MP_API_KEY`) and the `dft_3d` table of `jarvis-tools`; the cleaned tables in the preceding section are derived from them.

The NEMAD tables are not redistributed here because their redistribution terms were not verified. `nemad_ordering_validation.py` expects `data/nemad_classification_raw.csv` (columns `Normalized_Composition`, `Type`, ...; the ordering-classification table), and the Curie and Neel temperature tables `data/nemad_fm_curie_raw.csv` and `data/nemad_afm_neel_raw.csv`, downloaded from the Northeast Materials Database (Itani, Zhang and Zang, Nat. Commun. 16, 9415, 2025). The derived overlap and validation results are in `data/nemad_overlap_check.json` and `data/jarvis_ordering_validation.json`.

## Batch systems

`run_all.sh` is scheduler-independent. `example_hpc_job.sh` is a placeholder template for submitting it to a PBS cluster; replace every `<PLACEHOLDER>` with your own values.

## Requirements

Python 3.11 and the package versions in `requirements.txt` (PyTorch is needed for the domain-adversarial network only).

## License

MIT (see `LICENSE`). The underlying databases keep their own terms of use.
