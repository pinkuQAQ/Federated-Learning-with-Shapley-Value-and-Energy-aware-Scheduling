#!/bin/bash

set -eo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/zhaozhanshan/FLSV}
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/save"

RUN_TAG=${RUN_TAG:-significance_$(date +%Y%m%d_%H%M%S)}
MAX_PARALLEL=${MAX_PARALLEL:-8}
SUBMIT_ANALYSIS=${SUBMIT_ANALYSIS:-0}

ARRAY_JOB_ID=$(sbatch --parsable \
    --array="0-43%${MAX_PARALLEL}" \
    --export="ALL,RUN_TAG=${RUN_TAG},PROJECT_ROOT=${PROJECT_ROOT}" \
    scripts/slurm/run_significance_array.sh)

echo "Array job submitted: ${ARRAY_JOB_ID}"
echo "Run tag: ${RUN_TAG}"
echo "Tasks: 44, maximum concurrent GPU jobs: ${MAX_PARALLEL}"
echo "Results: ${PROJECT_ROOT}/save/significance/${RUN_TAG}"

if [ "${SUBMIT_ANALYSIS}" = "1" ]; then
    ANALYSIS_JOB_ID=$(sbatch --parsable \
        --dependency="afterok:${ARRAY_JOB_ID}" \
        --export="ALL,RUN_TAG=${RUN_TAG},PROJECT_ROOT=${PROJECT_ROOT}" \
        scripts/slurm/run_significance_analysis.sh)
    echo "Dependent analysis job submitted: ${ANALYSIS_JOB_ID}"
fi

