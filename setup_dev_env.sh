#!/usr/bin/env bash
set -e

VENV_NAME="wbf"

python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo ""
echo "Environment '$VENV_NAME' is ready. Installed packages:"
pip freeze
