import fire
import urllib.request
from pathlib import Path


langs = ['english', 'spanish', 'german', 'chinese', 'french', 'russian', 'japanese', 'italian']
lang_codes = ['en', 'es', 'de', 'zh', 'fr', 'ru', 'ja', 'it']

def fetch_mldoc(url_prefix, mldoc_path="data/mldoc"):
    """ Fetch mldoc from server using basic auth
        url_prefix should point to mldoc stored as follow
        "https://user:passwd@server/path/[english|spanish...].[dev|test|train.[1000|2000|5000|10000]].csv"
    """
    def fetch(url, mldoc):
        mldoc.parent.mkdir(parents=True, exist_ok=True)
        print("fetching", url, mldoc)
        urllib.request.urlretrieve(url, mldoc)
    for lang,code in zip(langs, lang_codes):
        for size in [1000, 2000, 5000, 10000]:
            dir = Path(mldoc_path)/f"{code}-{size // 1000}"
            fetch(f"{url_prefix}/{lang}.dev.csv", dir / f"{code}.dev.csv")
            fetch(f"{url_prefix}/{lang}.test.csv", dir / f"{code}.test.csv")
            fetch(f"{url_prefix}/{lang}.train.{size}.csv", dir / f"{code}.train.csv")
            fetch(f"{url_prefix}/{lang}.train.10000.csv", dir / f"{code}.unsup.csv")

if __name__ == "__main__":
    fire.Fire(fetch_mldoc)
