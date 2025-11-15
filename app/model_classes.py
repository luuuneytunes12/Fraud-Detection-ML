"""
Shared model classes for the fraud detection API.
This module contains class definitions that need to be available
for both training and inference (pickle loading).
"""

import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Handle both relative and absolute imports for FraudNN
try:
    from src.models import FraudNN
except ImportError:
    import sys
    import os

    # Add parent directory to path so we can import src
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from src.models import FraudNN


class PreTrainedFraudNN(BaseEstimator, ClassifierMixin):
    """
    Wrapper class for the pre-trained FraudNN model to work with sklearn's stacking classifier.
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.threshold = 0.5
        self.model_ = None

    def fit(self, X, y=None):
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        hyperparams = checkpoint["hyperparameters"]

        self.model_ = FraudNN(**hyperparams)
        self.model_.load_state_dict(checkpoint["model_state_dict"])
        self.model_.eval()

        self.threshold = hyperparams["threshold"]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X_tensor = torch.tensor(X.astype("float32"), dtype=torch.float32)
        with torch.no_grad():
            logits = self.model_(X_tensor).squeeze()
            probs = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > self.threshold).astype(int)


# Set estimator type for sklearn compatibility
PreTrainedFraudNN._estimator_type = "classifier"
