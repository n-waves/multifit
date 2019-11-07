import pandas as pd, numpy as np
import fire
from pathlib import Path
from sys import stderr
from sklearn.model_selection import train_test_split
import re

def to_csv(df, path):
    df.to_csv(path, header=None, index=None)

def remove_rt(df):
    return df.assign(text=df.text.str.replace('^RT @anonymized_account ',''))

def remove_duplicates(df):
    exact = df[~df.duplicated('text')]
    prefixes = exact.text.map(lambda t: t.endswith('…') and exact.text.str.startswith(t[:-1]).sum()>1)
    return exact[~prefixes]

def cross_remove_duplicates(from_df, other_df):
    exact = from_df[~from_df.text.isin(other_df.text)]
    other_prefixes = other_df.text[other_df.text.str.endswith('…')].str[:-1]
    if len(other_prefixes):
        other_prefixes_re = re.compile('^'+'|'.join([f'({re.escape(t)})' for t in other_prefixes]))
    else:
        other_prefixes_re = re.compile('^$')
    prefixes = exact.text.map(lambda t:
            (t.endswith('…') and other_df.text.str.startswith(t[:-1]).any()) or
            other_prefixes_re.match(t) is not None
        )
    return exact[~prefixes]

def split(data_dir, dedup=False):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "pl.unsup.csv", header=None, names=["label", "text"])
    val_ratio = 0.1
    train = remove_rt(train)
    trn, val = train_test_split(train, test_size=val_ratio, random_state=12345, stratify=train.label)

    if dedup:
        trn = remove_duplicates(trn)
        val = remove_duplicates(val)
        val = cross_remove_duplicates(val, trn)
        l1, l2, l3 = len(remove_duplicates(train)), len(trn), len(val)
        if l1 != l2 + l3:
            print("Warning: some condition believed by me to be invariant is not hold")
            print(f"{l1} should be equal to {l2} + {l3} = {l2+l3}")


    to_csv(trn, data_dir / "pl.train.csv")
    to_csv(val, data_dir / "pl.dev.csv")

if __name__ == "__main__": fire.Fire(split)
