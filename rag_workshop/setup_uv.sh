#!/usr/bin/env bash
# Create or synchronize the reproducible local uv environment for the workshop.
# Run from the workspace root: ./rag_workshop/setup_uv.sh

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v uv >/dev/null 2>&1; then
    UV=(uv)
elif command -v conda >/dev/null 2>&1 && conda run -n "${UV_CONDA_ENV:-ml}" uv --version >/dev/null 2>&1; then
    UV=(conda run -n "${UV_CONDA_ENV:-ml}" uv)
else
    echo "uv is not installed on this laptop." >&2
    echo "Install it once using the official installer, then rerun this script:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

"${UV[@]}" python pin 3.12
"${UV[@]}" sync --extra embeddings --extra secure-app
"${UV[@]}" run python -m ipykernel install --user \
    --name labobots-rag-workshop \
    --display-name "Python (LaboBots RAG workshop)"

echo "Local uv environment ready."
echo "Select the kernel: Python (LaboBots RAG workshop)"
echo "Run commands with: uv run <command>"
