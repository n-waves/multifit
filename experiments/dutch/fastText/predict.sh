#!/usr/bin/env sh

TEST_FILE=$1

./fasttext test \
	model.bin \
	$TEST_FILE 1
