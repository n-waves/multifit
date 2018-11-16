#!/usr/bin/env bash

ROOT="data"
DATA_DIR="${ROOT}/wiki/"
mkdir -p "${DATA_DIR}"
echo "Saving data in ""$DATA_DIR"
wget -c "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip" -P "${DATA_DIR}"
wget -c "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip" -P  "${DATA_DIR}"

unzip "${DATA_DIR}/wikitext-2-v1.zip" -d "${DATA_DIR}"
unzip "${DATA_DIR}/wikitext-103-v1.zip" -d "${DATA_DIR}"

for f in ${DATA_DIR}/wikitext-*/wiki.*.tokens; do
    nf=$(dirname $f)/en.$(basename $f)
    echo "Renaming $f to $nf"
    mv $f $(dirname $f)/en.$(basename $f)
done

echo "Please note wikitext en is already tokenized with Moses"