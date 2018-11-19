import zipfile
from pathlib import Path
from typing import Optional, Union

import fire
from tqdm import tqdm

from fastai.core import *
from fastai.datasets import *

ROOT = Path("data").resolve()
XNLI_DIR = ROOT / "xnli"
if not ROOT.exists():
    ROOT.mkdir()
XNLI_DIR.mkdir(exist_ok=True)

print(f"Saving data in {ROOT}")
MT_FILE = "XNLI-MT-1.0.zip"
XNLI_FILE = "XNLI-1.0.zip"
MT_PATH = XNLI_DIR / MT_FILE
XNLI_PATH = XNLI_DIR / XNLI_FILE
MT_URL = "https://s3.amazonaws.com/xnli/XNLI-MT-1.0.zip"
XNLI_URL = "https://s3.amazonaws.com/xnli/XNLI-1.0.zip"



class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_data(url: str, fname: Union[str, Path], dest: Optional[Union[str, Path]]):
    """
    Download data if the filename does not exist already
    Uses Tqdm to show download progress
    """
    from urllib.request import urlretrieve

    filepath = (Path(dest) / fname).resolve()

    if not filepath.exists():
        dirname = Path(filepath.parents[0])
        print(f"Creating directory {dirname} from {filepath}")
        dirname.mkdir(exist_ok=True)

        with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
            urlretrieve(url, filepath, reporthook=t.update_to)

    return str(filepath.resolve().absolute())


def get_and_unzip_data(url: str, fname: Union[str, Path] = None, dest: Union[str, Path] = None):
    """Download `url` if it doesn't exist to `fname` and un-tgz to folder `dest`"""
    if dest is None:
        dest = url.split("/")[-1]
    dest = Path(dest)
    fname = dest / fname
    if not fname.exists():
        download_data(url=url, fname=fname, dest=dest)
    print(f"Extracting {fname.resolve().absolute()} \n to {dest}")
    zipfile.ZipFile(fname, "r").extractall(dest)
    return dest


def get_xnli_and_MT(dest: Union[str, Path] = XNLI_DIR):
    get_and_unzip_data(url=XNLI_URL, fname=XNLI_FILE, dest=dest)
    get_and_unzip_data(url=MT_URL, fname=MT_FILE, dest=dest)


if __name__ == "__main__":
    fire.Fire(get_xnli_and_MT)
