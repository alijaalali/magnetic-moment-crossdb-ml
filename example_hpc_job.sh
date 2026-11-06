#!/bin/bash
# Example batch-job wrapper. Replace every <PLACEHOLDER> with the values of your own
# scheduler and cluster (PBS directives shown; adapt to Slurm if needed), then submit it.
#PBS -N magmom_pipeline
#PBS -A <YOUR_ALLOCATION>
#PBS -q <YOUR_QUEUE>
#PBS -l select=1:ncpus=<N_CPUS>:mem=<MEMORY>
#PBS -l walltime=<HH:MM:SS>
set -euo pipefail
cd "<PATH_TO_THIS_REPOSITORY>"
source <PATH_TO_YOUR_PYTHON_ENVIRONMENT>/bin/activate
bash run_all.sh
