import numpy as np


class DataChecker:

    RANDOM_STATE = 42
    N_CHANNELS = 128
    N_BANDS = 6
    N_TYPES = 2  # 0=abs, 1=rel
    N_FEATURES = N_CHANNELS * N_BANDS * N_TYPES

    @staticmethod
    def check_inputs(X, y):
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert X.ndim == 2, f"X must be 2D, got {X.ndim}D"
        assert y.ndim == 1, f"y must be 1D, got {y.ndim}D"
        assert X.shape[0] == y.shape[0], "X and y must have same n_samples"
        assert X.shape[1] == DataChecker.N_FEATURES, f"Expected {DataChecker.N_FEATURES} features, got {X.shape[1]}"
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf")
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or inf")
        classes = np.unique(y)
        assert set(classes).issubset({0, 1}), f"y must be binary 0/1, got {classes}"

        counts = {int(c): int((y == c).sum()) for c in classes}
        print("Input OK:", X.shape, y.shape, "class counts:", counts)

if __name__ == '__main__':
    import joblib
    data_path = '../../data/processed/test.pkl'

    dataset = joblib.load(data_path)
    datasets = dataset['datasets']

    rest_data = datasets[0]
    task_data = datasets[1]
    rest_labels = np.ones(rest_data.shape[0])*0
    task_labels = np.ones(task_data.shape[0])*1

    DataChecker.check_inputs(rest_data,rest_labels)
    DataChecker.check_inputs(task_data,task_labels)
