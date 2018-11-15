#!/usr/bin/env bash

ROOT="data"
DATA_DIR="${ROOT}/imdb"
mkdir -p "${DATA_DIR}"
echo "Saving data in $DATA_DIR"
wget -c "http://files.fast.ai/data/aclImdb.tgz" -P "${DATA_DIR}"

echo "Imdb is raw text so we are tokenizing it with Moses"
python -m fastai_contrib.utils prepare_imdb "${DATA_DIR}/aclImdb.tgz"

