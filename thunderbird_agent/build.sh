#!/usr/bin/env bash
# Package the extension as an installable .xpi (a plain zip with manifest.json at its root).
# Usage: ./build.sh   ->   dist/labobots-mail-agent-<version>.xpi
set -euo pipefail
cd "$(dirname "$0")"

version=$(python3 -c 'import json; print(json.load(open("manifest.json"))["version"])')
out="dist/labobots-mail-agent-${version}.xpi"

mkdir -p dist
rm -f "$out"
zip -q -r -X "$out" manifest.json background.js icons popup options vendor
echo "Built $out"
