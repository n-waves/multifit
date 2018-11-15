"""
Script to create small and large WikiText datasets from Wikipedia articles in
any language that were downloaded with `prepare_wiki.sh`.
Articles are tokenized using the Moses tokenizer. Articles with least than
100 tokens are removed.
"""
import argparse
from pathlib import Path
from collections import Counter
import json

from shutil import copyfile

from sacremoses import MosesTokenizer
from fastai_contrib.utils import replace_number, UNK
from fastai_contrib.tokenizers import get_sentencepiece, SentencepieceTokenizer


def get_texts(root):
    for dir_ in root.iterdir():
        for wiki_file in dir_.iterdir():
            with open(wiki_file, encoding='utf-8') as f_in:
                for line in f_in:
                    article = json.loads(line)
                    text = article['text']
                    title = article['title']
                    if text.strip() == title:
                        # print('No content continuing...')
                        continue
                    yield text


def write_wikitext(file_path, text_iter, tok, num_tokens, mode='w'):
    
    total_num_tokens = 0
    print(f'Writing to {file_path}...')
    i = 0
    
    with open(file_path, mode, encoding='utf-8') as f_out:
        for i, text in enumerate(text_iter):

            num_tokens_article = 0  # count the number of tokens in an article
            tokenized_paragraphs = []
            paragraphs = text.split('\n')

            for paragraph in paragraphs:
                tokenized = tok.tokenize(paragraph.strip(), return_str=True)
                tokenized_paragraphs.append(tokenized)

                tokens = tokenized.split(' ')  # split on whitespace to keep newlines
                # don't count empty lines
                tokens = [token for token in tokens if token]

                # calculate length based on tokens; add 1 for newline
                num_tokens_article += len(tokens) + 1

            if num_tokens_article < 100:
                # only use articles that have at least 100 tokens
                continue

            for tokenized in tokenized_paragraphs:
                f_out.write(tokenized + '\n')

            total_num_tokens += num_tokens_article + 1
            if total_num_tokens > num_tokens:
                break
            if i % 10000 == 0 and i > 0:
                print('Processed {:,} documents. Total # tokens: {:,}.'.format(i, total_num_tokens))
    print('{}. # documents: {:,}. # tokens: {:,}.'.format(
        file_path, i, total_num_tokens))

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
    print(f'{unk_path.name}. # of tokens: {total_num_tokens}')
    temp_file_path.replace(unk_path)

def replace_numbers(text_iter, unk_path):
    """
    Replace numbers as in Smerity's script:
    https://gist.github.com/Smerity/94af5902aa9498817c92d1e71eb2f87b#file-post_process-py
    :param file_path:
    :param unk_path:
    :return:
    """
    print(f'Replacing numbers in file. Writing to {unk_path}.')
    with open(unk_path, 'w', encoding='utf-8') as f:
        for text in text_iter:
            raw_tokens = line.strip().split(' ')
            tokens = []
            for token in raw_tokens:
                tokens.append(replace_number(token))
            # Starting each line with a blank line is required
            # Some systems replace \n with <eos> and assume, like in PTB, everything is space separated
            tokens = [''] + tokens + ['\n']
            line = ' '.join(tokens)
            f.write(line)


def main(args):

    input_path = Path(args.input)
    output = Path(args.output)
    assert input_path.exists(), f'Error: {input_path} does not exist.'
    output.mkdir(exist_ok=True)

    if args.subword:
        #  TO DO load the text corpus
        #  TO DO make get_sentencepiece return path to spm model
        spm_path = get_sentencepiece(output, corpus)
        tok = SentencepieceTokenizer(spm_path)
    else:
        tok = MosesTokenizer(args.lang)

    sml_wiki = output / f'{args.lang}-2'
    lrg_wiki = output / f'{args.lang}-100'
    sml_wiki.mkdir(exist_ok=True)
    lrg_wiki.mkdir(exist_ok=True)

    text_iter = get_texts(input_path)

    splits = ['train', 'valid', 'test']
    token_nums = [2000000, 200000, 200000]
    for split, token_num in zip(splits, token_nums):
        # TO DO maybe replace the numbers before tokenizing
        unk_path = wiki / f'{args.lang}.wiki.{split}.tokens.unk'
        replace_numbers(text_iter, unk_path)

        sml_file_path = sml_wiki / f'{args.lang}.wiki.{split}.tokens'
        write_wikitext(sml_file_path, text_iter, tok, token_num)
        lrg_file_path = lrg_wiki / f'{args.lang}.wiki.{split}.tokens'

        # copy the content of the small file to the large file
        print(f'Copying {sml_file_path} to {lrg_file_path}.')
        copyfile(sml_file_path, lrg_file_path)

    sml_vocab = build_vocab(sml_wiki_train)
    print(f'{args.lang}-2 vocab size: {len(sml_vocab)}')
    lrg_vocab = build_vocab(lrg_wiki_train)
    print(f'{args.lang}-100 vocab size: {len(lrg_vocab)}')

    # add the new articles to the existing ones
    lrg_wiki_train = lrg_wiki / f'{args.lang}.wiki.train.tokens'
    write_wikitext(lrg_wiki_train, text_iter, tok, 98000000, mode='a')

    # replace words not in the vocab with <unk>
    if not args.subword:
        for wiki, vocab in zip([sml_wiki, lrg_wiki], [sml_vocab, lrg_vocab]):
            for split in splits:
                unk_path = wiki / f'{args.lang}.wiki.{split}.tokens.unk'
                limit_vocab(unk_path, vocab)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='the directory where the Wikipedia data extracted '
                             'with WikiExtractor.py is located. Consists of '
                             'directories AA, AB, AC, etc.')
    parser.add_argument('-o', '--output', required=True,
                        help='the output directory where the merged Wikipedia '
                             'documents should be saved')
    parser.add_argument('-l', '--lang', required=True,
                        help='the iso code of the language of the Wikipedia '
                             'documents, e.g. en, fr, de, etc.')
    parser.add_argument('-sw', '--subword', default=False,
                        help='set to use sub-word tokenization (sentencepiece)'
                              'default tokenization method is Moses.')
    args = parser.parse_args()
    main(args)
