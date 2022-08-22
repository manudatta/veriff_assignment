from functools import lru_cache
from io import StringIO
from urllib.request import urlopen

import  tensorflow_hub as hub
from tensorflow_hub import KerasLayer


@lru_cache
def load_model(model_url: str) -> KerasLayer:
    return hub.KerasLayer(model_url)

@lru_cache
def download_url(url: str) -> StringIO:
    raw_data = urlopen(url)
    return raw_data
