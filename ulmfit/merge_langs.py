"""
Script to merge WikiText files created with `create_wikitext.py`.
"""
import fire
from pathlib import Path
from contextlib import ExitStack

def merge_wikitext(paths, langs, dest_path, num_sentences):
    wiki_paths = [Path(path) for path in paths]
    for wiki_path in wiki_paths:
        assert wiki_path.exists(), f'Error: {wiki_path} does not exist.'
    dest_path = Path(dest_path)
    dest_path.mkdir(exist_ok=True)
    splits = ['train', 'valid', 'test']
    concat_langs = '-'.join(langs)
    for split in splits:
        with ExitStack() as stack:
            files = [stack.enter_context(open(
                wiki_path / f'{lang}.wiki.{split}.tokens', 'r', encoding='utf-8')) 
                for lang, wiki_path in zip(langs, wiki_paths)]
            
            output = stack.enter_context(open(dest_path / f'{concat_langs}.wiki.{split}.tokens', 'w', encoding='utf-8'))
            done = False
            while not done:
                for file in files:
                    lines = [file.readline() for x in range(num_sentences)]
                    size = len(lines)
                    lines = [line for line in lines if line]
                    if len(lines) < size:
                        done = True
                    for line in lines:
                        output.write(line)

if __name__ == '__main__':
    fire.Fire(merge_wikitext)