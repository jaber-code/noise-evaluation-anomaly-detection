import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score

from enums import AggregationType


class AnomalyEvaluator:
    """
    Evaluates the trained NoiseEvaluator and compares against sklearn baselines.
    Returns AUC scores as a dict.
    """

    def __init__(
        self,
        model: nn.Module,
        agg_type: AggregationType,
        device: torch.device = torch.device("cpu"),
        logger=None,

    ):
        self.model = model
        self.agg_type = agg_type
        self.device = device
        self.logger = logger

    def _log(self, line: str) -> None:
        if self.logger is not None:
            self.logger.write(line)
        else:
            print(line)
    def _aggregate(self, abs_noise: torch.Tensor) -> torch.Tensor:
        """Equation 10 — aggregates per-feature noise predictions to a scalar score."""
        if self.agg_type == AggregationType.MAX:
            return torch.max(abs_noise, dim=1).values
        elif self.agg_type == AggregationType.MEAN:
            return torch.mean(abs_noise, dim=1)
        elif self.agg_type == AggregationType.MIN:
            return torch.min(abs_noise, dim=1).values
        elif self.agg_type == AggregationType.SUM:
            return torch.sum(abs_noise, dim=1)
        else:
            raise ValueError(f"Unsupported aggregation type: {self.agg_type}")

    def predict_scores(self, X: torch.Tensor) -> np.ndarray:
        """Anomaly scores — higher = more anomalous."""
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            predicted_noise = self.model(X)
            abs_noise = torch.abs(predicted_noise)
            scores = self._aggregate(abs_noise)
        return scores.cpu().numpy()

    def run(
        self,
        dataset_name: str,
        X_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: np.ndarray,
    ) -> dict:
        """Run our method + baselines on the same data. Returns AUCs."""
        self._log(f"\n--- Evaluating on {dataset_name} ---")


        # 1. Our method
        t0 = time.time()
        scores_ours = self.predict_scores(X_test)
        auc_ours = roc_auc_score(y_test, scores_ours)
        t_ours = time.time() - t0

        # Baselines need numpy on CPU
        X_train_np = X_train.cpu().numpy()
        X_test_np = X_test.cpu().numpy()

        # 2. IsolationForest
        t0 = time.time()
        iforest = IsolationForest(random_state=42).fit(X_train_np)
        # decision_function: higher = more normal. Negate for anomaly score.
        auc_if = roc_auc_score(y_test, -iforest.decision_function(X_test_np))

        #iforest = IsolationForest(n_estimators=100, random_state=42, contamination='auto').fit(X_train_np)
        # Use score_samples and negate (higher score_samples = more normal in sklearn)
        # auc_if = roc_auc_score(y_test, -iforest.score_samples(X_test_np))
        t_if = time.time() - t0

        # 3. LocalOutlierFactor (novelty mode for prediction on unseen test data)
        t0 = time.time()
        lof = LocalOutlierFactor(novelty=True).fit(X_train_np)
        auc_lof = roc_auc_score(y_test, -lof.decision_function(X_test_np))
        t_lof = time.time() - t0

        results = {
            "dataset": dataset_name,
            "Ours": auc_ours,
            "IsolationForest": auc_if,
            "LOF": auc_lof,
            "time_ours_sec": t_ours,
            "time_iforest_sec": t_if,
            "time_lof_sec": t_lof,
        }

        self._log(
            f"AUC — Ours: {auc_ours:.4f} | "
            f"IForest: {auc_if:.4f} | "
            f"LOF: {auc_lof:.4f}"
        )
        return results