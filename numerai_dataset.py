
from kedro.io.core import (
    AbstractDataSet
)

import tempfile
import pandas as pd
import numerapi
import requests, zipfile
from kedro_work.utils import get_joblib_memory

memory = get_joblib_memory()

@memory.cache
def download_url(url):
    r = requests.get(url)
    return r.content

class NumeraiDataset(AbstractDataSet):
    def __init__(self, is_train):
        self._is_train = is_train
        self._napi = numerapi.NumerAPI(verbosity="info")

    def _load(self):
        url = self._napi.get_dataset_url()

        with tempfile.TemporaryDirectory() as dir:
            cache_path = '{}/numerai_cache.zip'.format(dir)
            with open(cache_path, 'wb') as f:
                f.write(download_url(url))
            z = zipfile.ZipFile(cache_path)

            if self._is_train:
                fname = 'numerai_training_data.csv'
            else:
                fname = 'numerai_tournament_data.csv'

            df = pd.read_csv(z.open(fname), index_col=0)
        return df

    def _describe(self):
        return dict(is_train=self._is_train)

    def _save(self, data) -> None:
        pass
