#!/usr/bin/env bash

if [[ $# -ne 3 ]] ; then
  echo 'Usage: ./prepare_rcv.sh <RCV_URL> <RCV_USER> <RCV_PASSWORD>'
  echo 'This dataset has restricted access: apply at https://trec.nist.gov/data/reuters/reuters.html'
  exit 1
fi

ROOT="data"
DATA_DIR="${ROOT}/rcv"
mkdir -p "${DATA_DIR}/tmp"
echo "Saving data in $DATA_DIR"

RCV_URL=$1
RCV_USER=$2
RCV_PASSWORD=$3

MLDOC=https://github.com/facebookresearch/MLDoc/raw/master
wget -c $MLDOC/generate_documents.py -P "${DATA_DIR}/tmp"

if [ ! -d "${DATA_DIR}/tmp/RCV2_Multilingual_Corpus" ]; then
  wget -c --user $RCV_USER --password $RCV_PASSWORD $RCV_URL/rcv2.tar.xz -P "${DATA_DIR}/tmp"  
  tar xvf "${DATA_DIR}/tmp/rcv2.tar.xz" -C "${DATA_DIR}/tmp/"
else
  echo "RCV2 already exists. Skipping download."
fi

if [ ! -d "${DATA_DIR}/tmp/RCV2_Multilingual_Corpus/english" ]; then
  wget -c --user $RCV_USER --password $RCV_PASSWORD $RCV_URL/rcv1.tar.xz -P "${DATA_DIR}/tmp"
  tar xvf "${DATA_DIR}/tmp/rcv1.tar.xz" -C "${DATA_DIR}/tmp/RCV2_Multilingual_Corpus/"
  mv "${DATA_DIR}/tmp/RCV2_Multilingual_Corpus/rcv1" "${DATA_DIR}/tmp/RCV2_Multilingual_Corpus/english"
else
  echo "RCV1 already exists. Skipping download."
fi

for LANGUAGE in spanish chinese french japanese german italian russian english
do
  mkdir -p "${DATA_DIR}/${LANGUAGE}"
  for FILE_EXT in train.1000 train.2000 train.5000 train.10000 dev test
  do
    wget -c $MLDOC/mldoc-indices/$LANGUAGE.$FILE_EXT -P "${DATA_DIR}/tmp"

    python $DATA_DIR/tmp/generate_documents.py \
        --indices-file $DATA_DIR/tmp/$LANGUAGE.$FILE_EXT \
        --output-filename $DATA_DIR/tmp/$LANGUAGE.$FILE_EXT.raw \
        --rcv-dir $DATA_DIR/tmp/RCV2_Multilingual_Corpus/$LANGUAGE
    python ulmfit/postprocess_rcv.py --input_file $DATA_DIR/tmp/$LANGUAGE.$FILE_EXT.raw \
        --output_file $DATA_DIR/$LANGUAGE/$FILE_EXT.csv
  done
done
