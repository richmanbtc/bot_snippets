
from kedro.io.core import (
    AbstractDataSet
)

import pandas as pd
import numerapi

class NumeraiDataset2(AbstractDataSet):
    def __init__(self, is_train):
        self._is_train = is_train
        self._napi = numerapi.NumerAPI(verbosity="info")

    def _load(self):
        url = self._get_dataset_url()
        df = pd.read_parquet(url)
        return df

    def _get_current_round(self):
        return self._napi.get_current_round(tournament=8)

    def _get_dataset_url(self):
        round = self._get_current_round()

        filename = 'numerai_training_data.parquet' if self._is_train else 'numerai_tournament_data.parquet'

        query = """
            query ($filename: String!) {
                dataset(filename: $filename)
            }
            """
        params = {
            'filename': filename
        }
        if round:
            query = """
                        query ($filename: String!, $round: Int) {
                            dataset(filename: $filename, round: $round)
                        }
                        """
            params['round'] = round
        return self._napi.raw_query(query, params)['data']['dataset']

    def _describe(self):
        return dict(is_train=self._is_train)

    def _save(self, data) -> None:
        pass
