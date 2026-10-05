#!/usr/bin/env bash
# Package the extension as an installable .xpi (a plain zip with manifest.json at its root).
# Usage: ./build.sh   ->   dist/labobots-mail-agent-<version>.xpi
set -euo pipefail
cd "$(dirname "$0")"

# Check des dépendances
tools=('zip' 'python3')
for i in "${!tools[@]}" ; do type -fP "${tools[i]}" &> /dev/null || { >&2 printf "%b\n" "[ERROR] ${tools[i]} is missing"; exit 1; }; done

version=$(python3 -c 'import json; print(json.load(open("manifest.json"))["version"])')
out="dist/labobots-mail-agent-${version}.xpi"

mkdir -p dist
rm -f "$out"
zip -q -r -X "$out" manifest.json background.js rag.js streaming.js icons popup options vendor ||{ >&2 printf "%b\n" "[ERROR] creating ${out} failed" ; exit 1; };
echo "Built $out"
