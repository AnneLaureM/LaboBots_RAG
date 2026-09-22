#!/usr/bin/env bash
# Emergency operator tool for the LaboBots distributed RAG workshop.
# Run from the workspace root: ./rag_workshop/manage_remote_rag.sh <command>

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

REMOTE_USER="${REMOTE_USER:-labobots}"
REMOTE_HOST="${REMOTE_HOST:-195.221.220.18}"
REMOTE_PORT="${REMOTE_PORT:-22003}"
REMOTE_ROOT="${REMOTE_ROOT:-}"
REMOTE_VENV="${REMOTE_VENV:-}"
REMOTE_CHROMA_PORT="${REMOTE_CHROMA_PORT:-8000}"
REMOTE_LITELLM_PORT="${REMOTE_LITELLM_PORT:-4000}"
LOCAL_CHROMA_PORT="${LOCAL_CHROMA_PORT:-8000}"
LOCAL_LITELLM_PORT="${LOCAL_LITELLM_PORT:-4000}"
LOCAL_DB="${LOCAL_DB:-./rag_workshop/chroma_db}"
LOCAL_PROJECT="${LOCAL_PROJECT:-$SCRIPT_DIR/..}"
LOCAL_CONDA_ENV="${LOCAL_CONDA_ENV:-ml}"
LOCAL_LOG="${LOCAL_LOG:-/tmp/labobots-rag-tunnel.log}"
LOCAL_PID="${LOCAL_PID:-/tmp/labobots-rag-tunnel.pid}"

TARGET="${REMOTE_USER}@${REMOTE_HOST}"
SSH=(ssh -p "$REMOTE_PORT" -o ConnectTimeout=10 -o ServerAliveInterval=30)
SSH_FORWARD=(ssh -p "$REMOTE_PORT" -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes)
RSYNC_SSH="ssh -p $REMOTE_PORT -o ConnectTimeout=10"

usage() {
    cat <<'EOF'
Usage:
  manage_remote_rag.sh doctor
  manage_remote_rag.sh prepare
  manage_remote_rag.sh copy [--force]
  manage_remote_rag.sh start
  manage_remote_rag.sh stop
  manage_remote_rag.sh restart
  manage_remote_rag.sh status
  manage_remote_rag.sh verify [--all-ids]
  manage_remote_rag.sh tunnel [--chroma-port PORT] [--litellm-port PORT]
  manage_remote_rag.sh stop-tunnel
  manage_remote_rag.sh unlock

Environment overrides:
  REMOTE_USER, REMOTE_HOST, REMOTE_PORT, REMOTE_ROOT, REMOTE_VENV
  REMOTE_CHROMA_PORT, REMOTE_LITELLM_PORT
  LOCAL_CHROMA_PORT, LOCAL_LITELLM_PORT, LOCAL_DB

Examples:
  ./rag_workshop/manage_remote_rag.sh prepare
  ./rag_workshop/manage_remote_rag.sh copy --force
  ./rag_workshop/manage_remote_rag.sh tunnel --chroma-port 18000 --litellm-port 14000
  ./rag_workshop/manage_remote_rag.sh stop-tunnel
EOF
}

remote() {
    "${SSH[@]}" "$TARGET" "$@"
}

resolve_remote_paths() {
    local remote_home
    if [[ -z "$REMOTE_ROOT" || -z "$REMOTE_VENV" ]]; then
        remote_home="$(remote 'printf %s "$HOME"')"
        REMOTE_ROOT="${REMOTE_ROOT:-$remote_home/rag_workshop}"
        REMOTE_VENV="${REMOTE_VENV:-$remote_home/chroma-venv}"
    fi
    echo "Remote workspace: $REMOTE_ROOT"
}

require_local_db() {
    [[ -d "$LOCAL_DB" ]] || { echo "Missing local Chroma directory: $LOCAL_DB" >&2; exit 1; }
    [[ -f "$LOCAL_DB/chroma.sqlite3" ]] || { echo "Missing $LOCAL_DB/chroma.sqlite3" >&2; exit 1; }
}

local_chroma_version() {
    if command -v uv >/dev/null 2>&1 && [[ -f "$LOCAL_PROJECT/pyproject.toml" ]]; then
        uv run --project "$LOCAL_PROJECT" python -c 'import chromadb; print(chromadb.__version__)' 2>/dev/null && return 0
    fi
    local python_cmd
    python_cmd="$(command -v python || command -v python3 || true)"
    if [[ -n "$python_cmd" ]]; then
        "$python_cmd" -c 'import chromadb; print(chromadb.__version__)' 2>/dev/null && return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        conda run -n "$LOCAL_CONDA_ENV" python -c 'import chromadb; print(chromadb.__version__)' 2>/dev/null && return 0
    fi
    return 0
}

remote_process_pids() {
    # The bracket expression prevents pgrep from matching the pgrep command itself.
    remote "pgrep -u \"\$(id -u)\" -f '[c]hroma run.*--port $REMOTE_CHROMA_PORT' || true"
}

copy_database() {
    local force="${1:-0}"
    local restart_after_copy=0
    require_local_db
    local lock_dir="$REMOTE_ROOT/.chroma-operator.lock"
    local staging="$REMOTE_ROOT/.staging/chroma_db"
    local backup="$REMOTE_ROOT/backups/chroma_db-$(date +%Y%m%d-%H%M%S)"

    if ! remote "mkdir -p '$REMOTE_ROOT' && mkdir '$lock_dir'"; then
        echo "Remote operator lock exists: $lock_dir" >&2
        echo "Check that no other operation is active before removing it." >&2
        exit 1
    fi
    cleanup_lock() {
        remote "rmdir '$lock_dir' 2>/dev/null || true" >/dev/null 2>&1 || true
    }
    trap cleanup_lock RETURN

    if [[ "$force" != "1" ]] && remote "test -f '$REMOTE_ROOT/chroma_db/chroma.sqlite3'"; then
        echo "Remote database already exists; nothing copied. Use 'copy --force' to replace it."
        return
    fi

    if [[ "$force" == "1" ]] && remote "test -d '$REMOTE_ROOT/chroma_db'"; then
        if [[ -n "$(remote_process_pids)" ]]; then
            echo "Stopping remote Chroma before replacing its database."
            stop_remote
            restart_after_copy=1
        fi
        echo "Creating remote backup: $backup"
        remote "mkdir -p '$REMOTE_ROOT/backups' && mv '$REMOTE_ROOT/chroma_db' '$backup'"
    fi

    echo "Copying local Chroma database to remote staging..."
    remote "rm -rf '$staging' && mkdir -p '$staging'"
    rsync -az --info=progress2 -e "$RSYNC_SSH" "$LOCAL_DB/" "$TARGET:$staging/"
    remote "test -f '$staging/chroma.sqlite3' && mv '$staging' '$REMOTE_ROOT/chroma_db' && rmdir '$REMOTE_ROOT/.staging' 2>/dev/null || true"
    echo "Remote database copied."
    if [[ "$restart_after_copy" == "1" ]]; then
        start_remote
    fi
}

ensure_remote_venv() {
    local version
    version="$(local_chroma_version)"
    if remote "test -x '$REMOTE_VENV/bin/chroma'"; then
        echo "Remote Chroma environment already exists."
        return
    fi
    echo "Creating remote Chroma environment..."
    if remote "command -v uv >/dev/null 2>&1"; then
        remote "uv venv '$REMOTE_VENV'"
        if [[ -n "$version" ]]; then
            remote "uv pip install --python '$REMOTE_VENV/bin/python' 'chromadb==$version'"
        else
            remote "uv pip install --python '$REMOTE_VENV/bin/python' chromadb"
        fi
        echo "Remote environment prepared with uv."
        return
    fi
    if ! remote "python3 -m venv '$REMOTE_VENV'"; then
        echo "Remote Python cannot create virtual environments." >&2
        echo "Install the matching venv package on the server, then rerun prepare:" >&2
        remote "python3 --version" >&2 || true
        echo "  sudo apt-get update && sudo apt-get install -y python3.12-venv" >&2
        echo "  rm -rf '$REMOTE_VENV'" >&2
        echo "  ./rag_workshop/manage_remote_rag.sh prepare" >&2
        exit 1
    fi
    remote "'$REMOTE_VENV/bin/python' -m pip install --upgrade pip"
    if [[ -n "$version" ]]; then
        remote "'$REMOTE_VENV/bin/python' -m pip install 'chromadb==$version'"
    else
        remote "'$REMOTE_VENV/bin/python' -m pip install chromadb"
    fi
}

start_remote() {
    remote "test -f '$REMOTE_ROOT/chroma_db/chroma.sqlite3'" || {
        echo "Remote database is missing. Run: $0 prepare" >&2
        exit 1
    }
    ensure_remote_venv
    if [[ -n "$(remote_process_pids)" ]]; then
        echo "Chroma is already running on remote port $REMOTE_CHROMA_PORT."
        return
    fi
    echo "Starting remote Chroma..."
    remote "mkdir -p '$REMOTE_ROOT' && nohup '$REMOTE_VENV/bin/chroma' run --host 127.0.0.1 --port '$REMOTE_CHROMA_PORT' --path '$REMOTE_ROOT/chroma_db' >'$REMOTE_ROOT/chroma.log' 2>&1 < /dev/null &"
    echo "Chroma started. Log: $REMOTE_ROOT/chroma.log"
}

stop_remote() {
    local pids
    pids="$(remote_process_pids)"
    if [[ -z "$pids" ]]; then
        echo "No Chroma process found."
        return
    fi
    echo "Stopping remote Chroma PID(s): $pids"
    remote "kill $pids"
}

port_in_use() {
    local port="$1"
    (command -v lsof >/dev/null && lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null) || \
        (command -v ss >/dev/null && ss -ltn "sport = :$port" 2>/dev/null | tail -n +2 | grep -q .)
}

tunnel() {
    local chroma_port="$LOCAL_CHROMA_PORT"
    local litellm_port="$LOCAL_LITELLM_PORT"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --chroma-port) chroma_port="$2"; shift 2 ;;
            --litellm-port) litellm_port="$2"; shift 2 ;;
            *) echo "Unknown tunnel option: $1" >&2; usage; exit 2 ;;
        esac
    done

    if [[ -f "$LOCAL_PID" ]] && kill -0 "$(cat "$LOCAL_PID")" 2>/dev/null; then
        echo "A tunnel is already running with PID $(cat "$LOCAL_PID")."
        return
    fi

    if port_in_use "$chroma_port" || port_in_use "$litellm_port"; then
        echo "A requested local port is already in use." >&2
        echo "Choose alternatives, for example: $0 tunnel --chroma-port 18000 --litellm-port 14000" >&2
        exit 1
    fi

    nohup "${SSH_FORWARD[@]}" -N \
        -L "127.0.0.1:${chroma_port}:127.0.0.1:${REMOTE_CHROMA_PORT}" \
        -L "127.0.0.1:${litellm_port}:127.0.0.1:${REMOTE_LITELLM_PORT}" \
        "$TARGET" >"$LOCAL_LOG" 2>&1 &
    echo $! > "$LOCAL_PID"
    echo "Tunnel started with PID $!: Chroma localhost:$chroma_port, LiteLLM localhost:$litellm_port"
    echo "Log: $LOCAL_LOG"
}

stop_tunnel() {
    if [[ ! -f "$LOCAL_PID" ]]; then
        echo "No tunnel PID file found: $LOCAL_PID"
        return
    fi
    local pid
    pid="$(cat "$LOCAL_PID")"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        echo "Tunnel stopped: PID $pid"
    else
        echo "Tunnel process $pid is not running."
    fi
    rm -f "$LOCAL_PID"
}

unlock() {
    echo "Removing operator lock only; use this after confirming no copy/prepare operation is active."
    remote "rmdir '$REMOTE_ROOT/.chroma-operator.lock'"
    echo "Remote operator lock removed."
}

status() {
    echo "Local tunnel:"
    if [[ -f "$LOCAL_PID" ]] && kill -0 "$(cat "$LOCAL_PID")" 2>/dev/null; then
        echo "  running, PID $(cat "$LOCAL_PID")"
    else
        echo "  stopped"
    fi
    echo
    echo "Remote Chroma:"
    remote "if test -f '$REMOTE_ROOT/chroma_db/chroma.sqlite3'; then echo '  database: present'; else echo '  database: missing'; fi; pids=\$(pgrep -u \"\$(id -u)\" -f '[c]hroma run.*--port $REMOTE_CHROMA_PORT' || true); if test -n \"\$pids\"; then echo \"  Chroma PID(s): \$pids\"; else echo '  Chroma: stopped'; fi"
    echo
    echo "Remote services are expected on Chroma :$REMOTE_CHROMA_PORT and LiteLLM :$REMOTE_LITELLM_PORT."
}

doctor() {
    command -v ssh >/dev/null || { echo "Missing ssh"; exit 1; }
    command -v rsync >/dev/null || { echo "Missing rsync"; exit 1; }
    require_local_db
    echo "Local checks: OK"
    remote "echo 'SSH: OK'; command -v python3 >/dev/null && echo 'python3: OK'; if command -v sinfo >/dev/null; then sinfo -h -p labobots -o '%P %a %l' | head -n 1; else echo 'sinfo: unavailable on SSH host'; fi"
    status
}

verify() {
    local python_cmd
    if command -v uv >/dev/null 2>&1 && [[ -f "$LOCAL_PROJECT/pyproject.toml" ]]; then
        uv run --project "$LOCAL_PROJECT" python "$SCRIPT_DIR/verify_remote_chroma.py" \
            --local-db "$SCRIPT_DIR/chroma_db" \
            --remote-db "$REMOTE_ROOT/chroma_db" \
            --ssh-host "$REMOTE_HOST" --remote-user "$REMOTE_USER" --ssh-port "$REMOTE_PORT" "$@"
        return
    fi
    if command -v conda >/dev/null 2>&1 && conda run -n "$LOCAL_CONDA_ENV" python -c 'import chromadb' >/dev/null 2>&1; then
        conda run -n "$LOCAL_CONDA_ENV" python "$SCRIPT_DIR/verify_remote_chroma.py" \
            --local-db "$SCRIPT_DIR/chroma_db" \
            --remote-db "$REMOTE_ROOT/chroma_db" \
            --ssh-host "$REMOTE_HOST" --remote-user "$REMOTE_USER" --ssh-port "$REMOTE_PORT" "$@"
        return
    fi
    python_cmd="$(command -v python || command -v python3 || true)"
    if [[ -z "$python_cmd" ]]; then
        echo "Neither python nor python3 is available on the laptop." >&2
        exit 1
    fi
    "$python_cmd" "$SCRIPT_DIR/verify_remote_chroma.py" \
        --local-db "$SCRIPT_DIR/chroma_db" \
        --remote-db "$REMOTE_ROOT/chroma_db" \
        --ssh-host "$REMOTE_HOST" \
        --remote-user "$REMOTE_USER" \
        --ssh-port "$REMOTE_PORT" "$@"
}

command="${1:-help}"
shift || true
case "$command" in
    help|-h|--help) usage; exit 0 ;;
esac
resolve_remote_paths
case "$command" in
    doctor) doctor ;;
    prepare) copy_database 0; start_remote ;;
    copy) [[ "${1:-}" == "--force" ]] && copy_database 1 || copy_database 0 ;;
    start) start_remote ;;
    stop) stop_remote ;;
    restart) stop_remote; start_remote ;;
    status) status ;;
    verify) verify "$@" ;;
    tunnel) tunnel "$@" ;;
    stop-tunnel) stop_tunnel ;;
    unlock) unlock ;;
    *) echo "Unknown command: $command" >&2; usage; exit 2 ;;
esac
