#!/usr/bin/env bash

set -euo pipefail


# ============================================================
# Configuration
# ============================================================

ENV_NAME="wisdom"

LAMBDAFORGE_REPO="https://github.com/simplelambda/LambdaForge.git"
LAMBDAFORGE_MIN_VERSION="0.13.0"

MINIFORGE_DIR="$HOME/miniforge3"

AUTO_YES=false
CONDA_INSTALLED_BY_SCRIPT=false

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LAMBDAFORGE_DIR="${ROOT_DIR}/LambdaForge"
SIBLING_LAMBDAFORGE_DIR="$(dirname "${ROOT_DIR}")/LambdaForge"
OBSOLETE_WISDOM_METADATA="${ROOT_DIR}/src/wisdom_protein.egg-info"

TEMP_DIR=""


# ============================================================
# Cleanup
# ============================================================

cleanup() {
    if [[ -n "${TEMP_DIR}" && -d "${TEMP_DIR}" ]]; then
        rm -rf "${TEMP_DIR}"
    fi
}

trap cleanup EXIT INT TERM


# ============================================================
# Helpers
# ============================================================

usage() {
    cat <<EOF
Usage:
    ./install.sh
    ./install.sh -y
    ./install.sh --yes

Options:
    -y, --yes     Automatically accept the default installation.
    -h, --help    Show this help.

Default automatic behavior:
    - Use existing Conda if available.
    - Install Miniforge if Conda is missing.
    - Reuse ./LambdaForge if present.
    - Otherwise reuse ../LambdaForge if present.
    - Otherwise clone LambdaForge into ./LambdaForge.
    - Create/update Conda environment: wisdom.
    - Install LambdaForge and WISDOM in editable mode.
EOF
}


confirm() {
    local message="$1"

    if [[ "${AUTO_YES}" == true ]]; then
        echo "${message} [auto: yes]"
        return 0
    fi

    while true; do
        read -r -p "${message} [y/N]: " answer

        case "${answer}" in
            [Yy]|[Yy][Ee][Ss])
                return 0
                ;;

            ""|[Nn]|[Nn][Oo])
                return 1
                ;;

            *)
                echo "Please answer y or n."
                ;;
        esac
    done
}


command_exists() {
    command -v "$1" >/dev/null 2>&1
}


is_lambdaforge_checkout() {
    local path="$1"

    [[ -f "${path}/pyproject.toml" ]] || return 1
    [[ -d "${path}/src/lambdaforge" ]] || return 1

    grep -q 'name = "lambdaforge"' "${path}/pyproject.toml"
}


normalize_path() {
    local path="$1"

    python3 - "${path}" <<'PY'
import os
import sys

print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}


exclude_internal_lambdaforge_from_git() {
    local path="$1"

    if [[ "${path}" != "${DEFAULT_LAMBDAFORGE_DIR}" ]]; then
        return
    fi

    if [[ ! -d "${ROOT_DIR}/.git" ]]; then
        return
    fi

    local exclude_file="${ROOT_DIR}/.git/info/exclude"
    local rule="/LambdaForge/"

    mkdir -p "$(dirname "${exclude_file}")"
    touch "${exclude_file}"

    if ! grep -Fxq "${rule}" "${exclude_file}"; then
        echo "${rule}" >> "${exclude_file}"
        echo "LambdaForge excluded from this local Git working tree."
    fi
}


# ============================================================
# Arguments
# ============================================================

while [[ $# -gt 0 ]]; do
    case "$1" in

        -y|--yes)
            AUTO_YES=true
            shift
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done


cd "${ROOT_DIR}"


# ============================================================
# Banner
# ============================================================

echo
echo "============================================"
echo "           WISDOM installation"
echo "============================================"
echo
echo "Repository:"
echo "  ${ROOT_DIR}"
echo
echo "Conda environment:"
echo "  ${ENV_NAME}"
echo


# ============================================================
# Find existing Conda
# ============================================================

find_conda() {

    if command_exists conda; then
        return 0
    fi

    local candidates=(
        "$HOME/miniforge3"
        "$HOME/miniconda3"
        "$HOME/anaconda3"
    )

    for candidate in "${candidates[@]}"; do

        if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then

            # shellcheck disable=SC1090
            source "${candidate}/etc/profile.d/conda.sh"

            if command_exists conda; then
                return 0
            fi
        fi

    done

    return 1
}


# ============================================================
# Install Miniforge
# ============================================================

install_miniforge() {

    local os
    local arch
    local installer_name
    local installer_url
    local installer_path

    case "$(uname -s)" in

        Linux)
            os="Linux"
            ;;

        Darwin)
            os="MacOSX"
            ;;

        *)
            echo "ERROR: Unsupported operating system:"
            echo "  $(uname -s)"
            exit 1
            ;;
    esac


    case "$(uname -m)" in

        x86_64|amd64)
            arch="x86_64"
            ;;

        arm64)
            if [[ "${os}" == "MacOSX" ]]; then
                arch="arm64"
            else
                arch="aarch64"
            fi
            ;;

        aarch64)
            arch="aarch64"
            ;;

        *)
            echo "ERROR: Unsupported CPU architecture:"
            echo "  $(uname -m)"
            exit 1
            ;;
    esac


    installer_name="Miniforge3-${os}-${arch}.sh"

    installer_url="https://github.com/conda-forge/miniforge/releases/latest/download/${installer_name}"

    TEMP_DIR="$(mktemp -d)"
    installer_path="${TEMP_DIR}/${installer_name}"


    echo
    echo "Conda was not found."
    echo
    echo "Miniforge can be installed without sudo in:"
    echo
    echo "  ${MINIFORGE_DIR}"
    echo

    if ! confirm "Download and install Miniforge?"; then
        echo "Installation cancelled."
        exit 0
    fi


    echo
    echo "Downloading Miniforge..."

    if command_exists curl; then

        curl \
            --fail \
            --location \
            --progress-bar \
            "${installer_url}" \
            --output "${installer_path}"

    elif command_exists wget; then

        wget \
            "${installer_url}" \
            -O "${installer_path}"

    else

        echo
        echo "ERROR: Neither curl nor wget is installed."
        exit 1
    fi


    echo
    echo "Installing Miniforge..."

    bash "${installer_path}" \
        -b \
        -p "${MINIFORGE_DIR}"


    # Load Conda into this script
    # shellcheck disable=SC1091
    source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"

    CONDA_INSTALLED_BY_SCRIPT=true

    echo
    echo "Miniforge installed successfully."
}


# ============================================================
# Prepare Conda
# ============================================================

if find_conda; then

    echo "Conda detected:"
    echo "  $(command -v conda)"
    echo
    conda --version
    echo

    if ! confirm "Use this Conda installation?"; then
        echo "Installation cancelled."
        exit 0
    fi

else

    install_miniforge

fi


# ============================================================
# Optional shell initialization
# ============================================================

if [[ "${CONDA_INSTALLED_BY_SCRIPT}" == true ]]; then

    SHELL_NAME="$(basename "${SHELL:-bash}")"

    echo
    echo "Conda can configure '${SHELL_NAME}' so that:"
    echo
    echo "  conda activate wisdom"
    echo
    echo "works in future terminals."
    echo

    if confirm "Initialize Conda for ${SHELL_NAME}?"; then

        conda init "${SHELL_NAME}"

    else

        echo
        echo "Skipping shell initialization."
        echo
        echo "You can do it later with:"
        echo
        echo "  conda init ${SHELL_NAME}"
    fi

fi


# ============================================================
# Check environment.yml
# ============================================================

if [[ ! -f "${ROOT_DIR}/environment.yml" ]]; then

    echo
    echo "ERROR: environment.yml was not found:"
    echo
    echo "  ${ROOT_DIR}/environment.yml"
    exit 1

fi


# ============================================================
# Create/update Conda environment
# ============================================================

ENV_EXISTS=false

if conda env list \
    | awk 'NF && $1 !~ /^#/ {print $1}' \
    | grep -Fxq "${ENV_NAME}"
then
    ENV_EXISTS=true
fi


if [[ "${ENV_EXISTS}" == true ]]; then

    echo
    echo "Conda environment '${ENV_NAME}' already exists."
    echo

    if confirm "Update it from environment.yml?"; then

        echo
        echo "Updating environment..."

        conda env update \
            --name "${ENV_NAME}" \
            --file "${ROOT_DIR}/environment.yml" \
            --prune

    else

        echo
        echo "Keeping existing environment."

    fi

else

    echo
    echo "The Conda environment '${ENV_NAME}' does not exist."
    echo

    if confirm "Create environment '${ENV_NAME}'?"; then

        echo
        echo "Creating environment..."

        conda env create \
            --name "${ENV_NAME}" \
            --file "${ROOT_DIR}/environment.yml"

    else

        echo "Installation cancelled."
        exit 0

    fi

fi


# Verify environment really exists

if ! conda env list \
    | awk 'NF && $1 !~ /^#/ {print $1}' \
    | grep -Fxq "${ENV_NAME}"
then

    echo
    echo "ERROR: Conda environment '${ENV_NAME}' was not created."
    exit 1

fi


# ============================================================
# Locate / download LambdaForge
# ============================================================

LAMBDAFORGE_DIR=""


if [[ "${AUTO_YES}" == true ]]; then

    # Non-interactive preference:
    #
    # 1. WISDOM/LambdaForge
    # 2. ../LambdaForge
    # 3. clone into WISDOM/LambdaForge

    if is_lambdaforge_checkout "${DEFAULT_LAMBDAFORGE_DIR}"; then

        LAMBDAFORGE_DIR="${DEFAULT_LAMBDAFORGE_DIR}"

        echo
        echo "Using LambdaForge checkout:"
        echo "  ${LAMBDAFORGE_DIR}"

    elif is_lambdaforge_checkout "${SIBLING_LAMBDAFORGE_DIR}"; then

        LAMBDAFORGE_DIR="${SIBLING_LAMBDAFORGE_DIR}"

        echo
        echo "Using LambdaForge checkout:"
        echo "  ${LAMBDAFORGE_DIR}"

    else

        LAMBDAFORGE_DIR="${DEFAULT_LAMBDAFORGE_DIR}"

        echo
        echo "LambdaForge not found."
        echo "Cloning automatically into:"
        echo
        echo "  ${LAMBDAFORGE_DIR}"

    fi

else

    echo
    echo "============================================"
    echo "              LambdaForge"
    echo "============================================"
    echo

    if confirm "Do you already have LambdaForge downloaded?"; then

        while true; do

            echo
            read -r -p "Path to LambdaForge: " input_path

            if [[ -z "${input_path}" ]]; then
                echo "Please enter the path to your LambdaForge checkout."
                continue
            fi

            input_path="$(normalize_path "${input_path}")"

            if is_lambdaforge_checkout "${input_path}"; then

                LAMBDAFORGE_DIR="${input_path}"
                break

            fi

            echo
            echo "That directory does not look like LambdaForge:"
            echo "  ${input_path}"
            echo
            echo "Expected:"
            echo "  pyproject.toml"
            echo "  src/lambdaforge/"
            echo

        done

    else

        echo
        echo "LambdaForge will be downloaded from:"
        echo
        echo "  ${LAMBDAFORGE_REPO}"
        echo
        echo "Choose where to store it."
        echo
        echo "Press ENTER to use:"
        echo
        echo "  ${DEFAULT_LAMBDAFORGE_DIR}"
        echo

        read -r -p "Installation path: " input_path

        if [[ -z "${input_path}" ]]; then
            LAMBDAFORGE_DIR="${DEFAULT_LAMBDAFORGE_DIR}"
        else
            LAMBDAFORGE_DIR="$(normalize_path "${input_path}")"
        fi

    fi

fi


# ============================================================
# Clone LambdaForge if necessary
# ============================================================

if ! is_lambdaforge_checkout "${LAMBDAFORGE_DIR}"; then

    echo
    echo "LambdaForge needs to be downloaded to:"
    echo
    echo "  ${LAMBDAFORGE_DIR}"
    echo

    if [[ "${AUTO_YES}" != true ]]; then

        if ! confirm "Clone LambdaForge there?"; then
            echo "Installation cancelled."
            exit 0
        fi

    fi


    # --------------------------------------------------------
    # Ensure Git exists
    # --------------------------------------------------------

    if ! command_exists git; then

        echo
        echo "Git is required to download LambdaForge."

        if confirm "Install Git through Conda?"; then

            conda install \
                --name base \
                --channel conda-forge \
                --yes \
                git

        else

            echo "Installation cancelled."
            exit 1

        fi

    fi


    # --------------------------------------------------------
    # Validate destination
    # --------------------------------------------------------

    if [[ -e "${LAMBDAFORGE_DIR}" ]]; then

        if [[ -d "${LAMBDAFORGE_DIR}" ]] \
            && [[ -z "$(ls -A "${LAMBDAFORGE_DIR}")" ]]
        then

            rmdir "${LAMBDAFORGE_DIR}"

        else

            echo
            echo "ERROR: Destination already exists and is not"
            echo "a valid LambdaForge checkout:"
            echo
            echo "  ${LAMBDAFORGE_DIR}"
            echo
            exit 1

        fi

    fi


    mkdir -p "$(dirname "${LAMBDAFORGE_DIR}")"


    echo
    echo "Cloning LambdaForge..."

    git clone \
        "${LAMBDAFORGE_REPO}" \
        "${LAMBDAFORGE_DIR}"

fi


# Final validation

if ! is_lambdaforge_checkout "${LAMBDAFORGE_DIR}"; then

    echo
    echo "ERROR: LambdaForge checkout is invalid:"
    echo
    echo "  ${LAMBDAFORGE_DIR}"
    exit 1

fi


echo
echo "LambdaForge source:"
echo "  ${LAMBDAFORGE_DIR}"


# Do not show nested checkout in WISDOM git status

exclude_internal_lambdaforge_from_git "${LAMBDAFORGE_DIR}"


# ============================================================
# Install LambdaForge
# ============================================================

echo
echo "LambdaForge will be installed in editable mode:"
echo
echo "  pip install -e \"${LAMBDAFORGE_DIR}\""
echo

if confirm "Install LambdaForge into '${ENV_NAME}'?"; then

    conda run \
        --name "${ENV_NAME}" \
        python -m pip install \
        -e "${LAMBDAFORGE_DIR}"

else

    echo "Installation cancelled."
    exit 0

fi


# ============================================================
# Validate LambdaForge version
# ============================================================

LAMBDAFORGE_VERSION="$(
    conda run \
        --name "${ENV_NAME}" \
        python -c \
        'import importlib.metadata; print(importlib.metadata.version("lambdaforge"))'
)"


echo
echo "LambdaForge version:"
echo "  ${LAMBDAFORGE_VERSION}"


if ! conda run \
    --name "${ENV_NAME}" \
    python - "${LAMBDAFORGE_MIN_VERSION}" <<'PY'
import sys
from importlib.metadata import version

from packaging.version import Version

installed = Version(version("lambdaforge"))
minimum = Version(sys.argv[1])

if installed < minimum:
    raise SystemExit(
        f"Incompatible LambdaForge version: {installed}. "
        f"Required: >={minimum}"
    )
PY
then

    echo
    echo "ERROR: LambdaForge does not satisfy WISDOM requirements."
    exit 1

fi


# ============================================================
# Install WISDOM
# ============================================================

echo
echo "============================================"
echo "                 WISDOM"
echo "============================================"
echo
echo "WISDOM will be installed in editable mode:"
echo
echo '  pip install -e ".[dev]"'
echo
echo "LambdaForge is already installed, so pip will"
echo "satisfy the pyproject.toml dependency locally."
echo

if confirm "Install WISDOM and its Python dependencies?"; then

    # WISDOM releases before 0.13 used the distribution name ``wisdom-protein``. Because editable
    # metadata lives inside ``src/``, pip can still discover that obsolete distribution after the
    # project was renamed to ``wisdom`` and then report its retired LambdaForge upper bound. Remove
    # only that historical distribution metadata before installing the current project.

    if [[ -d "${OBSOLETE_WISDOM_METADATA}" ]] || conda run \
        --name "${ENV_NAME}" \
        python -m pip show \
        wisdom-protein >/dev/null 2>&1
    then

        echo
        echo "Removing obsolete wisdom-protein distribution metadata..."

        conda run \
            --name "${ENV_NAME}" \
            python -m pip uninstall \
            --yes \
            wisdom-protein

        # Very old editable installs have no uninstall RECORD, so pip identifies the distribution
        # but cannot delete its source-local metadata. The explicit target below is fixed, generated
        # packaging state; no source package or current ``wisdom.egg-info`` directory is removed.

        if [[ -d "${OBSOLETE_WISDOM_METADATA}" ]]; then
            rm -rf -- "${OBSOLETE_WISDOM_METADATA}"
        fi

    fi

    conda run \
        --name "${ENV_NAME}" \
        python -m pip install \
        --upgrade pip

    conda run \
        --name "${ENV_NAME}" \
        python -m pip install \
        -e "${ROOT_DIR}[dev]"

else

    echo "Installation cancelled."
    exit 0

fi


# ============================================================
# Validation
# ============================================================

echo
echo "============================================"
echo "             Installation check"
echo "============================================"
echo

if confirm "Run installation checks?"; then

    echo
    echo "Python:"
    conda run \
        --name "${ENV_NAME}" \
        python --version


    echo
    echo "Python dependency consistency:"
    conda run \
        --name "${ENV_NAME}" \
        python -m pip check


    echo
    echo "LambdaForge:"
    conda run \
        --name "${ENV_NAME}" \
        lf --version


    echo
    echo "MMseqs2:"
    conda run \
        --name "${ENV_NAME}" \
        mmseqs version


    echo
    echo "Foldseek:"
    conda run \
        --name "${ENV_NAME}" \
        foldseek version


    echo
    echo "Biopython:"
    conda run \
        --name "${ENV_NAME}" \
        python -c \
        'import Bio; print(Bio.__version__)'


    echo
    echo "WISDOM import:"
    conda run \
        --name "${ENV_NAME}" \
        python -c \
        'import wisdom; print("WISDOM import OK")'

fi


# ============================================================
# Done
# ============================================================

echo
echo "============================================"
echo "        WISDOM installation complete"
echo "============================================"
echo
echo "Conda environment:"
echo
echo "  ${ENV_NAME}"
echo
echo "LambdaForge:"
echo
echo "  ${LAMBDAFORGE_DIR}"
echo


if [[ "${CONDA_INSTALLED_BY_SCRIPT}" == true ]]; then

    echo "If Conda was initialized during this installation,"
    echo "open a new terminal before activating the environment."
    echo

fi


echo "Activate WISDOM with:"
echo
echo "  conda activate ${ENV_NAME}"
echo

echo "Useful checks:"
echo
echo "  python --version"
echo "  lf --version"
echo "  mmseqs version"
echo "  foldseek version"
echo
