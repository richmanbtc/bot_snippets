import mlflow
import yaml
import matplotlib.pyplot as plt
import cloudpickle
import tempfile
import lzma

class MlflowPlot():
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        plt.figure()
        plt.style.use('seaborn-darkgrid')
        return None

    def __exit__(self, type, value, traceback):
        with tempfile.TemporaryDirectory() as dir:
            fname = '{}/{}'.format(dir, self.filename)
            plt.savefig(fname, bbox_inches='tight') # tightでlegendが収まるようになる
            plt.close('all')
            mlflow.log_artifact(fname)

def mlflow_plot(filename):
    return MlflowPlot(filename)

def mlflow_log_model(model, path):
    if not path.endswith('.xz'):
        raise Exception('mlflow_log_model path must end with .xz')

    data = cloudpickle.dumps(model)
    data = lzma.compress(data)
    with tempfile.TemporaryDirectory() as dir:
        fname = '{}/{}'.format(dir, path)
        with open(fname, 'wb') as f:
            f.write(data)
        mlflow.log_artifact(fname)

def mlflow_log_yaml(obj, path):
    with tempfile.TemporaryDirectory() as dir:
        fname = '{}/{}'.format(dir, path)
        with open(fname, "w") as f:
            yaml.dump(obj, f)
        mlflow.log_artifact(fname)

def mlflow_log_str(x, path):
    with tempfile.TemporaryDirectory() as dir:
        fname = '{}/{}'.format(dir, path)
        with open(fname, "w") as f:
            f.write(str(x))
        mlflow.log_artifact(fname)
