"""
Script to post-process WikiText files created with `create_wikitext.py`.
Creates additional files where words not in the training data are replaced
with <UNK> and numbers are modified with a regex.
"""
import argparse

from collections import Counter
from pathlib import Path

import fire

from .utils import replace_number, UNK


def build_vocab(file_path, cutoff=3):
    counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            tokens = line.strip().split(' ') + ['<eos>']
            counter.update(tokens)
    vocab = {}
    in_vocab_count = 0
    OOV_count = 0
    for token, count in counter.most_common():
        if count >= cutoff:
            vocab[token] = count
            in_vocab_count += count
        else:
            OOV_count += count
    print('OOV ratio: %.4f.' % (OOV_count / (in_vocab_count + OOV_count)))
    return vocab


def limit_vocab(unk_path, vocab):
    """
    https://gist.github.com/Smerity/94af5902aa9498817c92d1e71eb2f87b#file-limit_vocab-py
    :param unk_path:
    :param vocab:
    :return:
    """
    temp_file_path = unk_path.with_name(unk_path.name + '.temp')
    total_num_tokens = 0
    print(f'Limiting vocab in {unk_path}. Writing to {unk_path}.')
    with open(unk_path, 'r', encoding='utf-8') as f_in, open(temp_file_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            tokens = [x for x in line.strip().split(' ') if x]
            tokens = [token if token in vocab else UNK for token in tokens]
            # Ensures there's a space between tokens, including the last word,
            # newline, and the first word of the next line
            tokens = tokens + ['\n']
            total_num_tokens += len(tokens)
            tokens = [''] + tokens
            line = ' '.join(tokens)
            f_out.write(line)
    print(f'{unk_path.name}. # of tokens: {total_num_tokens}')
    temp_file_path.replace(unk_path)


def replace_numbers(file_path, unk_path):
    """
    Replace numbers as in Smerity's script:
    https://gist.github.com/Smerity/94af5902aa9498817c92d1e71eb2f87b#file-post_process-py
    :param file_path:
    :param unk_path:
    :return:
    """
    print(f'Replacing numbers in {file_path}. Writing to {unk_path}.')
    with open(file_path, 'r', encoding='utf-8') as f_in, open(unk_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            raw_tokens = line.strip().split(' ')
            tokens = []
            for token in raw_tokens:
                tokens.append(replace_number(token))
            # Starting each line with a blank line is required
            # Some systems replace \n with <eos> and assume, like in PTB, everything is space separated
            tokens = [''] + tokens + ['\n']
            line = ' '.join(tokens)
            f_out.write(line)


def postprocess_wikitext(path, lang):

    wiki_path = Path(path)
    assert wiki_path.exists(), f'Error: {wiki_path} does not exist.'
    dest_path = wiki_path.parent / (wiki_path.name + "-unk")
    dest_path.mkdir(exist_ok=True)
    splits = ['train', 'valid', 'test']
    for split in splits:
        # replace numbers with placeholders
        file_path = wiki_path / f'{lang}.wiki.{split}.tokens'
        assert file_path.exists(), f"Error: {file_path} does not exist."
        unk_path = dest_path / file_path.name
        replace_numbers(file_path, unk_path)

    # replace words not in the vocab with <unk>
    wiki_train = dest_path / f'{lang}.wiki.train.tokens'
    vocab = build_vocab(wiki_train)
    print(f'{wiki_path} vocab size: {len(vocab)}')
    for split in splits:
        unk_path = dest_path / f'{lang}.wiki.{split}.tokens'
        limit_vocab(unk_path, vocab)


if __name__ == '__main__':
    fire.Fire(postprocess_wikitext)
