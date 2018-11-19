import fire
import pandas as pd

def main(input_file, output_file):
  df = pd.read_csv(input_file, sep='\t', header=None)
  unique_labels = sorted(list(df[0].unique()))
  labels = [unique_labels.index(label) for label in df[0]]
  texts = [eval(text).decode('utf-8').strip() for text in df[1]]
  out = pd.DataFrame({'labels': labels, 'texts': texts})
  out.to_csv(output_file, header=None, index=False)

if __name__ == '__main__': fire.Fire(main)
