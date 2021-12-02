
from kedro.io.core import (
    AbstractDataSet
)

import joblib
from mlflow.tracking import MlflowClient
import tempfile

class MlflowArtifactDataset(AbstractDataSet):
    def __init__(self, run_id, artifact_path):
        self._run_id = run_id
        self._artifact_path = artifact_path

    def _load(self):
        with tempfile.TemporaryDirectory() as dest_path:
            client = MlflowClient()
            path = client.download_artifacts(
                run_id=self._run_id,
                path=self._artifact_path,
                dst_path=dest_path
            )
            return joblib.load(path)

    def _describe(self):
        return dict(run_id=self._run_id, artifact_path=self._artifact_path)

    def _save(self, data) -> None:
        pass
