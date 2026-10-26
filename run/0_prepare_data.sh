#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-all}"
DATA_DIR="${REPO_ROOT}/data"

case "${MODE}" in
    em|sycophancy|squad|all) ;;
    *)
        echo "Usage: $0 [em|sycophancy|squad|all]" >&2
        exit 2
        ;;
esac

if [[ "${MODE}" == "all" || "${MODE}" == "em" || "${MODE}" == "sycophancy" ]]; then
    python "${REPO_ROOT}/scripts/download_data.py" \
        --mode "${MODE/ all/all}" \
        --data-dir "${DATA_DIR}"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "sycophancy" ]]; then
    python "${REPO_ROOT}/scripts/prepare_data.py" \
        --mode sycophancy \
        --data-dir "${DATA_DIR}" \
        --out-dir "${REPO_ROOT}"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "em" ]]; then
    python "${REPO_ROOT}/scripts/prepare_data.py" \
        --mode em \
        --data-dir "${DATA_DIR}" \
        --out-dir "${REPO_ROOT}"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "squad" ]]; then
    python "${REPO_ROOT}/scripts/prepare_data.py" \
        --mode squad \
        --data-dir "${DATA_DIR}" \
        --out-dir "${REPO_ROOT}"
fi
