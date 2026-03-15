#!/usr/bin/env bash
# Task-analogy project build script.
# - Uses a Conda environment named `analogy_env` by default.
# - If the Conda env does not exist, creates it, then installs dependencies inside it.
# - If the Conda env already exists, just activates it and installs / runs kg/syn.
# - Ensures config (api.json) when missing.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-analogy_env}"
PYTHON="${PYTHON:-python}"  # used inside conda env

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 [install|kg|syn|full|help]"
    echo ""
    echo "  install   Create Conda env (if needed), activate, then install dependencies (default)."
    echo "  kg        Run AnalogyKG build (activate Conda env if exists)."
    echo "  syn       Run AnalogySyn (activate Conda env if exists)."
    echo "  full      install, then kg, then syn."
    echo "  help      Show this message."
    echo ""
    echo "Env: CONDA_ENV=$CONDA_ENV, PYTHON=$PYTHON"
    echo "     OPENAI_API_KEY used by both KG and Syn when set."
}

# ---------------------------------------------------------------------------
# Conda env: create if missing, then activate
# ---------------------------------------------------------------------------
activate_conda_env() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "Error: conda command not found. Please install Anaconda/Miniconda first."
        return 1
    fi
    # Initialize conda for this shell
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        echo "Activating existing Conda env: $CONDA_ENV"
        conda activate "$CONDA_ENV"
    else
        echo "Conda env '$CONDA_ENV' not found. Create it first: $0 install"
        return 1
    fi
}

# Create Conda env if it does not exist; then activate it.
ensure_conda_env() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "Error: conda command not found. Please install Anaconda/Miniconda first."
        exit 1
    fi
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        echo "Creating Conda env '$CONDA_ENV' ..."
        conda create -n "$CONDA_ENV" python=3.10 -y
    fi
    echo "Activating Conda env: $CONDA_ENV"
    conda activate "$CONDA_ENV"
}

# ---------------------------------------------------------------------------
# Install dependencies (always in Conda env: create+activate or just activate)
# ---------------------------------------------------------------------------
do_install() {
    ensure_conda_env
    echo "Installing AnalogyKG dependencies ..."
    pip install -q -r AnalogyKG/requirements.txt
    echo "Installing AnalogySyn dependencies ..."
    pip install -q -r AnalogySyn/requirements.txt
    echo "Install done."
}

# ---------------------------------------------------------------------------
# Ensure config exists (no overwrite)
# ---------------------------------------------------------------------------
ensure_config() {
    for dir in AnalogyKG AnalogySyn; do
        cfg="$dir/config/api.json"
        if [ ! -f "$cfg" ]; then
            echo "Warning: $cfg not found (please create it and set openai_api_key if needed)."
        fi
    done
}

# ---------------------------------------------------------------------------
# Run AnalogyKG build (activate existing Conda env, then run)
# ---------------------------------------------------------------------------
do_kg() {
    activate_conda_env || exit 1
    ensure_config
    echo "Running AnalogyKG build ..."
    (cd AnalogyKG && python build.py "$@")
}

# ---------------------------------------------------------------------------
# Run AnalogySyn (activate existing Conda env, then run)
# ---------------------------------------------------------------------------
do_syn() {
    activate_conda_env || exit 1
    ensure_config
    echo "Running AnalogySyn run ..."
    (cd AnalogySyn && python run.py "$@")
}

# ---------------------------------------------------------------------------
# Full pipeline: kg then syn
# ---------------------------------------------------------------------------
do_full() {
    do_install
    do_kg
    do_syn
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-install}" in
    install)
        do_install
        ensure_config
        echo ""
        echo "Next: set OPENAI_API_KEY and run ./build.sh kg and/or ./build.sh syn"
        ;;
    kg)
        do_kg "${@:2}"
        ;;
    syn)
        do_syn "${@:2}"
        ;;
    full)
        do_full
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
esac
