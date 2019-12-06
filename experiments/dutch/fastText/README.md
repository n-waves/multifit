# fastText classifier baseline

## Description

This folder contains scripts that were used to obtain a baseline for the sentiment polarity classification task.

## fastText

### Install

We'll be using the command-line tool, which supports using pre-trained word embeddings. Instructions for downloading and building fastText can be found here: https://github.com/facebookresearch/fastText

### Word embeddings

Pre-trained word embeddings for Dutch can be downloaded from: https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.nl.zip

Extract them to the current directory: `unzip wiki.nl.zip`

## Dataset

### Download

The experiments were run on 110kDBRD dataset, which can be downloaded from here: https://github.com/benjaminvdb/110kDBRD

### Convert

The 110kDBRD dataset is in a different format and needs to be converted first. Run `prepare.py` to convert the *extracted* dataset and save it to the current directory.

````
python ./prepare.py /path/to/110kDBRD
```` 

### Modelling

## Train

````
./train.sh train.txt ./wiki.nl.vec
Read 26M words
Number of words:  665350
Number of labels: 2
Progress: 100.0% words/sec/thread:  337040 lr:  0.000000 loss:  0.074446 ETA:   0h 0m
````

## Test

````
./predict.sh test.txt
N	10972
P@1	0.809
R@1	0.809
````