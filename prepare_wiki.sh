#!/usr/bin/env bash
# Script to download a Wikipedia dump

# Script is partially based on https://github.com/facebookresearch/fastText/blob/master/get-wikimedia.sh
ROOT="data"
echo "Saving data in ""$ROOT"

if [ "$1" == "" ] ; then
    read -r -p "Choose a language (e.g. en, bh, fr, etc.): " choice
    LANG="$choice"
else
    LANG="$1"
fi
echo "Chosen language: ""$LANG"

if [ "$2" == "" ] ; then
    read -p "Enter the minimal tokens per articles [100]: " tokens_min
    TOKENS_MIN=${tokens_min:-100}
else
    TOKENS_MIN="$2"
fi

DUMP_DIR="${ROOT}/wiki_dumps"
EXTR_DIR="${ROOT}/wiki_extr"
WIKI_DIR="${ROOT}/wiki"
EXTR="wikiextractor"
mkdir -p "${ROOT}"
mkdir -p "${DUMP_DIR}"
mkdir -p "${EXTR_DIR}"
mkdir -p "${WIKI_DIR}"

DUMP_FILE="${LANG}wiki-latest-pages-articles.xml.bz2"
DUMP_PATH="${DUMP_DIR}/${DUMP_FILE}"

if [ ! -f "${DUMP_PATH}" ]; then
  wget -c "https://dumps.wikimedia.org/""${LANG}""wiki/latest/""${DUMP_FILE}""" -P "${DUMP_DIR}"
else
  echo "${DUMP_PATH} already exists. Skipping download."
fi

# Check if directory exists
if [ ! -d "${EXTR}" ]; then
  git clone https://github.com/attardi/wikiextractor.git
  cd "${EXTR}"
  python setup.py install
  cd ..
fi

EXTR_PATH="${EXTR_DIR}/${LANG}"
if [ ! -d "${EXTR_PATH}" ]; then
  python wikiextractor/WikiExtractor.py -s --json -o "${EXTR_PATH}" "${DUMP_PATH}"
else
  echo "${EXTR_PATH} already exists. Skipping extraction."
fi

python -m ulmfit.create_wikitext -i "${EXTR_PATH}"  -l "${LANG}" -o "${WIKI_DIR}" -t "${TOKENS_MIN}"

python -m ulmfit.postprocess_wikitext "${WIKI_DIR}/${LANG}-2" $LANG
python -m ulmfit.postprocess_wikitext "${WIKI_DIR}/${LANG}-100" $LANG
#python -m ulmfit.postprocess_wikitext "${WIKI_DIR}/${LANG}-all" $LANG
