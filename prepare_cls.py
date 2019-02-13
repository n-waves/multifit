import fire
import urllib.request
from pathlib import Path

lang_codes = ['fr', 'ja', 'en', 'de']

def fetch_cls(url_prefix, cls_path="data/cls"):
    """ Fetch CLS from server using basic auth
        url_prefix should point to CLS stored as follow
        "https://user:passwd@server/path/[en|fr|de|jp]/[dvd|music|books].[test|train|unlabeled].csv"
    """
    def fetch(url, CLS):
        CLS.parent.mkdir(parents=True, exist_ok=True)
        print("fetching", url, CLS)
        urllib.request.urlretrieve(url, CLS)
    for code in lang_codes:
        for category in ['books', 'music', 'dvd']:
            dir = Path(cls_path)/f'{code}-{category}'
            fetch(f"{url_prefix}/{code}/{category}/train.csv", dir / f"{code}.train.csv")
            fetch(f"{url_prefix}/{code}/{category}/test.csv", dir / f"{code}.test.csv")
            fetch(f"{url_prefix}/{code}/{category}/unlabeled.csv", dir / f"{code}.unsup.csv")

if __name__ == "__main__":
    fire.Fire(fetch_cls)
