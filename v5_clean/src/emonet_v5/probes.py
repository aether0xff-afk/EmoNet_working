from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgeBinaryProbe:
    """Small deterministic linear probe with no sklearn dependency.

    The probe is deliberately simple. It standardizes features using training
    statistics and fits a ridge-regression classifier to targets {-1, +1}.
    For p >> n it uses the dual solution so trace vectors can remain raw and
    high-dimensional without a large feature-space matrix inversion.
    """

    alpha: float = 1.0
    eps: float = 1e-6

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "RidgeBinaryProbe":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if x.ndim != 2:
            raise ValueError("features must have shape [samples, features]")
        if y.shape != (x.shape[0],):
            raise ValueError("labels must have shape [samples]")
        if x.shape[0] < 2:
            raise ValueError("at least two samples are required")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("labels must be binary 0/1")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")

        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < self.eps] = 1.0
        z = (x - self.mean_) / self.scale_
        target = y.astype(np.float64) * 2.0 - 1.0

        n_samples, n_features = z.shape
        if n_features > n_samples:
            gram = z @ z.T
            dual = np.linalg.solve(
                gram + self.alpha * np.eye(n_samples, dtype=np.float64),
                target,
            )
            self.weight_ = z.T @ dual
        else:
            gram = z.T @ z
            self.weight_ = np.linalg.solve(
                gram + self.alpha * np.eye(n_features, dtype=np.float64),
                z.T @ target,
            )
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "weight_"):
            raise RuntimeError("probe must be fitted before prediction")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.weight_.shape[0]:
            raise ValueError("feature shape does not match fitted probe")
        z = (x - self.mean_) / self.scale_
        return z @ self.weight_

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.decision_function(features) >= 0.0).astype(np.int64)

    def accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        pred = self.predict(features)
        if pred.shape != y.shape:
            raise ValueError("labels shape does not match predictions")
        return float(np.mean(pred == y))
