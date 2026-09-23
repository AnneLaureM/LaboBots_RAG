#!/usr/bin/env bash
# Administrator-only LiteLLM lifecycle and participant-key provisioning tool.
# Run from the laptop workspace root; never commit the generated key file.

set -Eeuo pipefail

REMOTE_USER="${REMOTE_USER:-labobots}"
REMOTE_HOST="${REMOTE_HOST:-195.221.220.18}"
REMOTE_PORT="${REMOTE_PORT:-22003}"
REMOTE_ROOT="${REMOTE_ROOT:-}"
REMOTE_LITELLM_PORT="${REMOTE_LITELLM_PORT:-4000}"
LITELLM_MODEL_NAME="${LITELLM_MODEL_NAME:-workshop-llm}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
REMOTE_MIN_FREE_GB="${REMOTE_MIN_FREE_GB:-10}"
TARGET="${REMOTE_USER}@${REMOTE_HOST}"
SSH=(ssh -p "$REMOTE_PORT" -o ConnectTimeout=10 -o ServerAliveInterval=30)

usage() {
    cat <<'EOF'
Usage:
  manage_litellm.sh status
  manage_litellm.sh install-db
  manage_litellm.sh start
  manage_litellm.sh stop
  manage_litellm.sh create-keys [--count 36] [--prefix participant] [--duration 8h] [--budget 5]

install-db installs PostgreSQL on the remote host and provisions a "litellm"
database + role, which LiteLLM's proxy needs to persist and generate
participant virtual keys. Run it once before the first create-keys.

It needs an account with sudo rights on the remote host, which does not have
to be the same account that runs LiteLLM (REMOTE_USER). If your low-privilege
account has no sudo, point install-db at your admin account instead:
  ADMIN_USER=myadmin ADMIN_HOST=host ADMIN_PORT=22 manage_litellm.sh install-db
(each defaults to REMOTE_USER/REMOTE_HOST/REMOTE_PORT when unset). The
generated database password is printed once to your terminal — copy it, then
paste it when 'start' prompts for it.

The master key and DB password are read interactively from the local terminal
and are never printed by 'start'. Participant keys are stored remotely with
mode 600 in:
  $HOME/rag_workshop/participant-keys.tsv
EOF
}

remote() { "${SSH[@]}" "$TARGET" "$@"; }

check_remote_disk() {
    local available_kb required_kb
    available_kb="$(remote "df -Pk '$REMOTE_ROOT' | awk 'NR==2 {print \$4}'")"
    required_kb=$((REMOTE_MIN_FREE_GB * 1024 * 1024))
    if [[ "$available_kb" -lt "$required_kb" ]]; then
        echo "Insufficient remote disk space: $((available_kb / 1024 / 1024)) GB available; ${REMOTE_MIN_FREE_GB} GB required." >&2
        echo "Clean /mnt/backup or lower REMOTE_MIN_FREE_GB only after checking the risk." >&2
        exit 1
    fi
    echo "Remote disk space: $((available_kb / 1024 / 1024)) GB available."
}

resolve_remote_paths() {
    if [[ -z "$REMOTE_ROOT" ]]; then
        local home
        home="$(remote 'printf %s "$HOME"')"
        REMOTE_ROOT="$home/rag_workshop"
    fi
    echo "Remote workspace: $REMOTE_ROOT"
}

prompt_master_key() {
    if [[ -n "${LITELLM_MASTER_KEY:-}" ]]; then
        MASTER_KEY="$LITELLM_MASTER_KEY"
    else
        read -r -s -p "LiteLLM master key (input hidden): " MASTER_KEY
        printf '\n' >&2
    fi
    [[ -n "$MASTER_KEY" ]] || { echo "Master key cannot be empty." >&2; exit 1; }
}

prompt_db_password() {
    if [[ -n "${LITELLM_DB_PASSWORD:-}" ]]; then
        DB_PASSWORD="$LITELLM_DB_PASSWORD"
    else
        read -r -s -p "LiteLLM Postgres password (leave empty to skip persistent virtual keys): " DB_PASSWORD
        printf '\n' >&2
    fi
}

install_db() {
    local admin_user="${ADMIN_USER:-$REMOTE_USER}"
    local admin_host="${ADMIN_HOST:-$REMOTE_HOST}"
    local admin_port="${ADMIN_PORT:-$REMOTE_PORT}"
    local db_role="$REMOTE_USER"
    echo "Connecting to ${admin_user}@${admin_host}:${admin_port} to install PostgreSQL (needs sudo there)."
    ssh -t -p "$admin_port" -o ConnectTimeout=10 "${admin_user}@${admin_host}" "
        set -e
        if ! command -v psql >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y postgresql
        else
            echo 'PostgreSQL already installed.'
        fi
        if sudo -u postgres psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='$db_role'\" | grep -q 1; then
            echo 'Role $db_role already exists in PostgreSQL; leaving it untouched.'
        else
            db_pass=\$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 32)
            sudo -u postgres psql -v ON_ERROR_STOP=1 -c \"CREATE USER $db_role WITH PASSWORD '\$db_pass';\" -c \"CREATE DATABASE litellm OWNER $db_role;\"
            echo
            echo '=========================================================='
            echo \"DB password for '$db_role' - copy it now, it will not be shown again:\"
            echo \"\$db_pass\"
            echo '=========================================================='
        fi
    "
    echo "Next: run 'manage_litellm.sh start' and paste that password when prompted (or Ctrl+C / leave it empty to skip persistent keys)."
}

status() {
    remote "if pgrep -u \"\$(id -u)\" -f '[l]itellm.*--port $REMOTE_LITELLM_PORT' >/dev/null; then echo 'LiteLLM: running'; else echo 'LiteLLM: stopped'; fi; if command -v ollama >/dev/null 2>&1; then echo 'Ollama: installed'; else echo 'Ollama: missing'; fi; df -h \"\$(dirname '$REMOTE_ROOT')\" | tail -n 1"
}

start() {
    check_remote_disk
    prompt_master_key
    prompt_db_password
    remote "command -v ollama >/dev/null 2>&1 || { echo 'Ollama is missing on the server.' >&2; exit 1; }; ollama list >/dev/null"
    remote "mkdir -p '$REMOTE_ROOT'"
    printf '%s\n%s\n' "$MASTER_KEY" "$DB_PASSWORD" | remote 'read -r MASTER_KEY; read -r DB_PASSWORD; umask 077; cat > "$HOME/rag_workshop/litellm_config.yaml" <<EOF
model_list:
  - model_name: workshop-llm
    litellm_params:
      model: ollama/'"$OLLAMA_MODEL"'
      api_base: http://127.0.0.1:11434

general_settings:
  master_key: "$MASTER_KEY"
EOF
if [[ -n "$DB_PASSWORD" ]]; then
    echo "  database_url: \"postgresql://'"$REMOTE_USER"':$DB_PASSWORD@localhost:5432/litellm\"" >> "$HOME/rag_workshop/litellm_config.yaml"
fi
'
    remote "if pgrep -u \"\$(id -u)\" -f '[l]itellm.*--port $REMOTE_LITELLM_PORT' >/dev/null; then echo 'LiteLLM already running'; else \
        if ! command -v uv >/dev/null 2>&1; then echo 'Installing uv for the labobots account...'; curl -LsSf https://astral.sh/uv/install.sh | sh; fi; \
        export PATH=\"\$HOME/.local/bin:\$HOME/litellm-venv/bin:\$PATH\"; \
        [ -d \"\$HOME/litellm-venv\" ] || uv venv \"\$HOME/litellm-venv\" >/dev/null; \
        uv pip install --python \"\$HOME/litellm-venv/bin/python\" 'litellm[proxy]' >/dev/null; \
        if grep -q '^  database_url:' \"$REMOTE_ROOT/litellm_config.yaml\"; then \
            echo 'Preparing Postgres-backed virtual keys (prisma client)...'; \
            uv pip install --python \"\$HOME/litellm-venv/bin/python\" prisma >/dev/null; \
            schema_path=\$(\"\$HOME/litellm-venv/bin/python\" -c \"import litellm, os; print(os.path.join(os.path.dirname(litellm.__file__), 'proxy', 'schema.prisma'))\"); \
            \"\$HOME/litellm-venv/bin/python\" -m prisma generate --schema=\"\$schema_path\" >/dev/null; \
        fi; \
        nohup \"\$HOME/litellm-venv/bin/litellm\" --config \"$REMOTE_ROOT/litellm_config.yaml\" --port '$REMOTE_LITELLM_PORT' --host 127.0.0.1 >\"$REMOTE_ROOT/litellm.log\" 2>&1 < /dev/null & echo \"LiteLLM PID: \$!\"; fi"
}

stop() {
    remote "pids=\$(pgrep -u \"\$(id -u)\" -f '[l]itellm.*--port $REMOTE_LITELLM_PORT' || true); if test -n \"\$pids\"; then kill \$pids; echo \"Stopped LiteLLM: \$pids\"; else echo 'LiteLLM already stopped'; fi"
}

create_keys() {
    local count=36 prefix=participant duration=8h budget=5
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --count) count="$2"; shift 2 ;;
            --prefix) prefix="$2"; shift 2 ;;
            --duration) duration="$2"; shift 2 ;;
            --budget) budget="$2"; shift 2 ;;
            *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
        esac
    done
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "Count must be a positive integer." >&2; exit 2; }
    [[ "$duration" =~ ^[1-9][0-9]*(s|m|h|d)$ ]] || { echo "Duration must look like 30s, 45m, 8h, or 20d." >&2; exit 2; }
    [[ "$budget" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Budget must be a positive number (USD)." >&2; exit 2; }
    prompt_master_key
    printf '%s\n' "$MASTER_KEY" | remote 'read -r MASTER_KEY; set -e; test -n "$(pgrep -u "$(id -u)" -f "[l]itellm.*--port '"$REMOTE_LITELLM_PORT"'" || true)" || { echo "LiteLLM is not running on port '"$REMOTE_LITELLM_PORT"'. Run start first." >&2; exit 1; }; grep -q "^  database_url:" "$HOME/rag_workshop/litellm_config.yaml" 2>/dev/null || { echo "No database configured (see litellm_config.yaml) -- create-keys needs a Postgres-backed LiteLLM. Run '"'"'install-db'"'"', then '"'"'start'"'"' again with the DB password." >&2; exit 1; }; umask 077; output="$HOME/rag_workshop/participant-keys.tsv"; : > "$output"; for i in $(seq 1 '"$count"'); do alias="'"$prefix"'-$(printf "%02d" "$i")"; curl -sS -X POST http://127.0.0.1:'"$REMOTE_LITELLM_PORT"'/key/delete -H "Authorization: Bearer $MASTER_KEY" -H "Content-Type: application/json" -d "{\"key_aliases\":[\"$alias\"]}" >/dev/null 2>&1 || true; response=$(curl -sS -w "\n%{http_code}" -X POST http://127.0.0.1:'"$REMOTE_LITELLM_PORT"'/key/generate -H "Authorization: Bearer $MASTER_KEY" -H "Content-Type: application/json" -d "{\"duration\":\"'"$duration"'\",\"max_budget\":'"$budget"',\"models\":[\"'"$LITELLM_MODEL_NAME"'\"],\"key_alias\":\"$alias\"}"); http_code=$(printf "%s" "$response" | tail -n1); body=$(printf "%s" "$response" | sed "\$d"); key=$(printf "%s" "$body" | python3 -c "import json,sys; print(json.load(sys.stdin).get(\"key\", \"\"))" 2>/dev/null || true); test "${key#sk-}" != "$key" || { echo "Key generation failed for $alias (HTTP $http_code): $body" >&2; exit 1; }; printf "%s\t%s\n" "$alias" "$key" >> "$output"; done; chmod 600 "$output"; echo "Generated '"$count"' keys: $output"'
}

command="${1:-help}"
shift || true
case "$command" in
    help|-h|--help) usage ;;
    status) resolve_remote_paths; status ;;
    install-db) install_db ;;
    start) resolve_remote_paths; start ;;
    stop) resolve_remote_paths; stop ;;
    create-keys) resolve_remote_paths; create_keys "$@" ;;
    *) echo "Unknown command: $command" >&2; usage; exit 2 ;;
esac
