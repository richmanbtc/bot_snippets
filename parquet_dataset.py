
from kedro.io.core import (
    AbstractDataSet
)

import pandas as pd

class ParquetDataset(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = filepath

    def _load(self):
        return pd.read_parquet(self._filepath)

    def _describe(self):
        return dict(filepath=self._filepath)

    def _save(self, data) -> None:
        pass
