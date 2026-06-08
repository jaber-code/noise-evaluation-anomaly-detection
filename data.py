import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.model_selection import train_test_split


class ADBenchDataLoader:
    """
    Loads ADBench Classical datasets from a local data/ directory.

    Datasets must be downloaded manually from:
    https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical
    and placed in the data/ directory (e.g. data/6_cardio.npz).
    """
    seeds = [0, 1, 2, 3, 4]

    def __init__(self, dataset_name: str, data_dir: str = "data", seed: int = 42):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.seed = seed

    def _load(self) -> tuple[np.ndarray, np.ndarray]:
        path = os.path.join(self.data_dir, f"{self.dataset_name}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset not found at {path}.\n"
                f"Download {self.dataset_name}.npz from "
                f"https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical "
                f"and place it in {self.data_dir}/"
            )

        data = np.load(path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        return X, y

    @staticmethod
    def _standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Standardize using training mean/std (paper protocol)."""
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0  # avoid div-by-zero for constant features
        return (X_train - mean) / std, (X_test - mean) / std

    def prepare_occ(self, test_size: float = 0.5) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        OCC protocol: 50% of normal samples for training,
        the rest of normals + all anomalies for testing.
        """
        X, y = self._load()
        normal_idx = np.where(y == 0)[0]
        anomaly_idx = np.where(y == 1)[0]

        train_idx, test_normal_idx = train_test_split(
            normal_idx, test_size=test_size, random_state=self.seed
        )

        X_train = X[train_idx]
        test_idx = np.concatenate([test_normal_idx, anomaly_idx])
        X_test = X[test_idx]
        y_test = y[test_idx]

        X_train, X_test = self._standardize(X_train, X_test)

        return (
            torch.from_numpy(X_train),
            torch.from_numpy(X_test),
            y_test,
        )

    def prepare_uad(self, test_size: float = 0.3) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        UAD protocol: stratified random split of full dataset.
        Anomalies appear in both train and test; labels ignored at train time.
        """
        X, y = self._load()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        #X_train = X_train[y_train == 0]
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        #X_train, X_test = self._standardize(X_train, X_test)

        return (
            torch.from_numpy(X_train),
            torch.from_numpy(X_test),
            y_test,
        )