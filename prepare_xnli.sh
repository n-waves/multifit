#!/usr/bin/env bash
# Script to download a Wikipedia dump

# Script is partially based on https://github.com/facebookresearch/fastText/blob/master/get-wikimedia.sh
ROOT="data"
XNLI_DIR="${ROOT}/xnli"
mkdir -p "${ROOT}"
mkdir -p "${XNLI_DIR}"

echo "Saving data in ""$ROOT"
MT_FILE="XNLI-MT-1.0.zip"
XNLI_FILE="XNLI-1.0.zip" 
MT_PATH="${XNLI_DIR}/${MT_FILE}"
XNLI_PATH="${XNLI_DIR}/${XNLI_FILE}"

if [ ! -f "${MT_PATH}" ]; then
  wget -c "https://s3.amazonaws.com/xnli/XNLI-MT-1.0.zip" -P "${XNLI_DIR}"
  wget -c "https://s3.amazonaws.com/xnli/XNLI-1.0.zip" -P "${XNLI_DIR}"
else
  echo "${MT_PATH} already exists. Skipping download."
fi

unzip "${MT_PATH}" -d "${XNLI_DIR}"
unzip "${XNLI_PATH}" -d "${XNLI_DIR}"

echo "Please note xnli en is already tokenized with Moses"
