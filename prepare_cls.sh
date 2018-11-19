#!/usr/bin/env bash

ROOT=data
DATA_DIR=$ROOT/cls
mkdir -p $DATA_DIR/tmp
echo "Saving data in $DATA_DIR"

if [ ! -d $DATA_DIR/tmp/cls-acl10-unprocessed ]; then
  wget -c http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-cls-10/cls-acl10-unprocessed.tar.gz  -P $DATA_DIR/tmp
  tar -xzvf $DATA_DIR/tmp/cls-acl10-unprocessed.tar.gz -C $DATA_DIR/tmp/
else
  echo "CLS already exists. Skipping download."
fi

python ulmfit/postprocess_cls.py --input_dir $DATA_DIR/tmp/cls-acl10-unprocessed --output_dir $DATA_DIR
