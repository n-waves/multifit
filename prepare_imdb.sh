#!/usr/bin/env bash

ROOT="data"
DATA_DIR="${ROOT}/imdb"
mkdir -p "${DATA_DIR}"
echo "Saving data in $DATA_DIR"
wget -c "http://files.fast.ai/data/aclImdb.tgz" -P "${DATA_DIR}"

echo "Imdb is raw text no preparation is done"
python -m multifit.datasets.utils prepare_imdb "${DATA_DIR}/aclImdb.tgz"

