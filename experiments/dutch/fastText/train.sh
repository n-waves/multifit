#!/usr/bin/env sh

TRAIN_FILE=$1
PRETRAINED=$2

./fasttext supervised \
	-input $TRAIN_FILE \
	-output model \
	-epoch 25 \
	-wordNgrams 4 \
	-dim 300 \
	-loss hs \
	-thread 7 \
	-minCount 1 \
	-lr 1.0 \
	-verbose 2 \
	-pretrainedVectors $PRETRAINED
