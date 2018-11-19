import fire
import pandas as pd
import os
from bs4 import BeautifulSoup

def main(input_dir, output_dir):
  for lang in ['en', 'de', 'fr', 'jp']:
    for cat in ['dvd', 'music', 'books']:
      for mode in ['train', 'test']: # , 'unlabeled']:
        os.makedirs(os.path.join(input_dir, lang), exist_ok=True)
        with open(os.path.join(input_dir, lang, cat, mode + '.review'), 'r') as f:
          items = BeautifulSoup(f.read(), features="html.parser").find_all('item')
          text = [item.find('text').text.strip() for item in items]
          summary = [item.find('summary').text.strip() for item in items]
          if mode == 'unlabeled':
            out = pd.DataFrame({'summary': summary, 'text': text})
          else:
            labels = [1 if item.rating.text in ('4.0', '5.0') else 0 for item in items]
            out = pd.DataFrame({'labels': labels, 'summary': summary, 'text': text})
          file_name = os.path.join(output_dir, f'{lang}/{cat}.{mode}.csv')
          out.to_csv(file_name, header=None, index=False)

if __name__ == '__main__': fire.Fire(main)
