#!/usr/bin/env bash -x
source /workspace/venv/bin/activate
PYTHON=${PYTHON:-"python"}
REPO="$( cd "$(dirname "$0")" ; cd .. ; pwd -P )"
cd $REPO
$PYTHON -m pip install -e .
$PYTHON -m pip install torch
$PYTHON -m pip install spacy
#$PYTHON -m spacy download en
$PYTHON -m pip install git+https://github.com/PiotrCzapla/sotabench-eval.git
