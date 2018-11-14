#!/usr/bin/env bash
# Script to download a Wikipedia dump

# Script is partially based on https://github.com/facebookresearch/fastText/blob/master/get-wikimedia.sh
ROOT="data"
DUMP_DIR="${ROOT}/xnli_dnld"
EXTR_DIR="${ROOT}/xnli_extr"
WIKI_DIR="${ROOT}/xnli"
EXTR="wikiextractor"
mkdir -p "${ROOT}"
mkdir -p "${DUMP_DIR}"
mkdir -p "${WIKI_DIR}"

echo "Saving data in ""$ROOT"
read -r -p "Choose a language (e.g. en, fr, etc.): " choice
LANG="$choice"
echo "Chosen language: ""$LANG"
MT_FILE="XNLI-MT-1.0.zip"
XNLI_FILE="XNLI-1.0.zip" 
MT_PATH="${DUMP_DIR}/${MT_FILE}"
XNLI_PATH="${DUMP_DIR}/${XNLI_FILE}"

if [ ! -f "${MT_PATH}" ]; then
  read -r -p "Continue to download (WARNING: This might be big and can take a long time!) (y/n)? " choice
  case "$choice" in
    y|Y ) echo "Starting download...";;
    n|N ) echo "Exiting";exit 1;;
    * ) echo "Invalid answer";exit 1;;
  esac
  wget -c "https://s3.amazonaws.com/xnli/XNLI-MT-1.0.zip" -P "${DUMP_DIR}"
  wget -c "https://s3.amazonaws.com/xnli/XNLI-1.0.zip" -P "${DUMP_DIR}"

else
  echo "${MT_PATH} already exists. Skipping download."
fi

if [ ! -d "${EXTR_DIR}" ]; then
  read -r -p "Continue to extract XNLI (WARNING: This might take a long time!) (y/n)? " choice
  case "$choice" in
    y|Y ) echo "Extracting ${MT_PATH} to ${EXTR_DIR}...";;
    n|N ) echo "Exiting";exit 1;;
    * ) echo "Invalid answer";exit 1;;
  esac
  unzip "${MT_PATH}" -d "${EXTR_DIR}"
  unzip "${XNLI_PATH}" -d "${EXTR_DIR}"
else
  echo "${EXTR_DIR} already exists. Skipping extraction."
fi
