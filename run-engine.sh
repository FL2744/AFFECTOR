#!/bin/zsh
set -e
cd "${0:A:h}"
RES="$PWD/dist/AFFECTOR.app/Contents/Resources"
export PYTHONHOME="$RES/Python.framework/Versions/3.14"
export PYTHONPATH="$RES/packages"
export HF_HOME="$RES/engine/.hf"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec "$PYTHONHOME/bin/python3.14" -u engine/bridge.py
