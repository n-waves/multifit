#!/usr/bin/env python2

import os
import sys
import codecs
import re
import tarfile

from sklearn.datasets import load_files


def convert(input_dir, output_file):
    """
    Convert 110kDBRD dataset into fastText compatible format.
    """
    regex = re.compile(r'\s+')
    dataset = load_files(input_dir, encoding='utf-8')
    with codecs.open(output_file, 'w', encoding='utf-8') as f:
        buff = u'\n'.join([u'__label__{} {}'.format(target, regex.sub(' ', text).strip()) for target, text in zip(dataset.target, dataset.data)])
        f.write(buff)


def main():
    """
    Expects the root of 110kDBRD as input argument. Converts and saves to ./train.txt and ./test.txt
    """
    base_dir = sys.argv[1]
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    convert(train_dir, 'train.txt')
    convert(test_dir, 'test.txt')


if __name__ == '__main__':
    main()