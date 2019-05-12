import pandas as pd, numpy as np
import fire
from pathlib import Path
from sys import stderr
from sklearn.model_selection import train_test_split

def to_csv(df, path):
    df.to_csv(path, header=None, index=None)

def split(data_dir):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "pl.unsup.csv", header=None)
    trn, val = train_test_split(train, test_size=0.1, random_state=12345, stratify=train[0])

    to_csv(trn, data_dir / "pl.train.csv")
    to_csv(val, data_dir / "pl.dev.csv")

if __name__ == "__main__": fire.Fire(split)
