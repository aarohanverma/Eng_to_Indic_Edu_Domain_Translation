#!/bin/bash
# -----------------------------------------------------------------------------
# setup.sh
#
# This script sets up the project by unzipping required dependency archives,
# installing necessary packages, building extensions, and running additional
# installation scripts. It assumes that the dependencies are stored in the
# "dependencies" folder of the project.
#
# The script follows DRY and KISS principles by using helper functions and
# loops where appropriate.
#
# Usage: ./setup.sh
# -----------------------------------------------------------------------------

set -e  # Exit immediately if a command exits with a non-zero status

# -----------------------------------------------------------------------------
# Resolve the project root directory (one level above the directory of this script)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# -----------------------------------------------------------------------------
# Define dependency directories and paths
# -----------------------------------------------------------------------------
DEPS_DIR="$PROJECT_ROOT/dependencies"
INDIC_NLP_LIB_DIR="$DEPS_DIR/indic_nlp_library"
INDIC_TRANS_TOOLKIT_DIR="$DEPS_DIR/IndicTransToolkit"
INDIC_TRANS2_DIR="$DEPS_DIR/IndicTrans2"

echo "Base project path: $PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Function: run_command
# Executes a command string and prints a message.
# If the command fails, prints an error message and exits.
#
# Arguments:
#   $1 - The command string to execute.
#   $2 - The error message to display if the command fails.
# -----------------------------------------------------------------------------
run_command() {
    echo "-> $1"
    eval "$1"
    if [ $? -ne 0 ]; then
        echo "Error: $2"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Extract dependency zip files using a loop.
# -----------------------------------------------------------------------------
echo "Extracting dependencies..."
zip_files=("indic_nlp_library.zip" "IndicTransToolkit.zip" "IndicTrans2.zip")
for zip_file in "${zip_files[@]}"; do
    run_command "unzip -o $DEPS_DIR/$zip_file -d $DEPS_DIR" "Failed to unzip $zip_file"
done

# -----------------------------------------------------------------------------
# Install IndicTransToolkit and its dependencies.
# -----------------------------------------------------------------------------
echo "Installing IndicTransToolkit..."
run_command "cd $INDIC_NLP_LIB_DIR && pip install --editable ./" "Failed to install indic_nlp_library"
run_command "cd $INDIC_TRANS_TOOLKIT_DIR && pip install -r requirements.txt && pip install --editable . --use-pep517" "Failed to install toolkit requirements"
run_command "cd $INDIC_TRANS_TOOLKIT_DIR && python3 setup.py build_ext --inplace" "Build failed"

# Install additional pip packages listed in the project's requirements.txt.
run_command "cd $DEPS_DIR && pip install -r requirements.txt" "Failed to install extra pip packages"

# -----------------------------------------------------------------------------
# Download NLTK punkt_tab data.
# -----------------------------------------------------------------------------
echo "Downloading NLTK punkt_tab data..."
python -c "import nltk; nltk.download('punkt_tab')"

# -----------------------------------------------------------------------------
# Run the install script for IndicTrans2 from the huggingface_interface directory.
# -----------------------------------------------------------------------------
echo "Running install script for IndicTrans2..."
run_command "cd $INDIC_TRANS2_DIR/huggingface_interface && bash install.sh" "IndicTrans2 install script failed"

# -----------------------------------------------------------------------------
# Uninstall and then reinstall IndicTransToolkit to ensure the latest build is used.
# -----------------------------------------------------------------------------
run_command "pip uninstall -y IndicTransToolkit" "Uninstall failed"
run_command "pip install -e $INDIC_TRANS_TOOLKIT_DIR --use-pep517" "Re-install failed"
