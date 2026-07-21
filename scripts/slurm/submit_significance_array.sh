#!/bin/bash

set -eo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/zhaozhanshan/FLSV}
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/save"

RUN_TAG=${RUN_TAG:-significance_$(date +%Y%m%d_%H%M%S)}
WORKER_COUNT=${WORKER_COUNT:-4}
MAX_PARALLEL=${MAX_PARALLEL:-${WORKER_COUNT}}
SUBMIT_ANALYSIS=${SUBMIT_ANALYSIS:-0}

if ! [[ "${WORKER_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: WORKER_COUNT must be a positive integer"
    exit 1
fi
if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_PARALLEL must be a positive integer"
    exit 1
fi

LAST_WORKER=$((WORKER_COUNT - 1))

ARRAY_JOB_ID=$(sbatch --parsable \
    --array="0-${LAST_WORKER}%${MAX_PARALLEL}" \
    --export="ALL,RUN_TAG=${RUN_TAG},PROJECT_ROOT=${PROJECT_ROOT},WORKER_COUNT=${WORKER_COUNT}" \
    scripts/slurm/run_significance_array.sh)

echo "Array job submitted: ${ARRAY_JOB_ID}"
echo "Run tag: ${RUN_TAG}"
echo "Worker jobs: ${WORKER_COUNT}, maximum concurrent GPU jobs: ${MAX_PARALLEL}"
echo "Experiments: 44, approximately $(((44 + WORKER_COUNT - 1) / WORKER_COUNT)) per worker"
echo "Results: ${PROJECT_ROOT}/save/significance/${RUN_TAG}"

if [ "${SUBMIT_ANALYSIS}" = "1" ]; then
    ANALYSIS_JOB_ID=$(sbatch --parsable \
        --dependency="afterok:${ARRAY_JOB_ID}" \
        --export="ALL,RUN_TAG=${RUN_TAG},PROJECT_ROOT=${PROJECT_ROOT}" \
        scripts/slurm/run_significance_analysis.sh)
    echo "Dependent analysis job submitted: ${ANALYSIS_JOB_ID}"
fi
