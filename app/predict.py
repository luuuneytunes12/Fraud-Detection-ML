# First, import the required modules and make the class available in the right namespace
import joblib
import numpy as np
import os
import sys

# Import the class and add it to __main__ module so pickle can find it
# Handle both relative and absolute imports
try:
    from app.model_classes import PreTrainedFraudNN
except ImportError:
    from model_classes import PreTrainedFraudNN

import __main__

__main__.PreTrainedFraudNN = PreTrainedFraudNN

# Now load the stacking classifier - handle path based on current working directory
if os.path.exists("models/stacking_classifier.pkl"):
    model_path = "models/stacking_classifier.pkl"
else:
    model_path = "../models/stacking_classifier.pkl"

stacking_clf = joblib.load(model_path)
print("Model loaded successfully!")


def make_prediction(features) -> dict:
    """
    Make prediction using the stacking classifier.

    Parameters
    ----------
    features : InputFeatures (Pydantic model)
        Input features as a Pydantic model with named fields

    Returns
    -------
    dict
        Prediction result with probability and class
    """
    # Convert Pydantic model to numpy array in the correct order
    # Order: Time, V1-V28, Amount (matching dataset column order)
    feature_values = [
        features.Time,
        features.V1,
        features.V2,
        features.V3,
        features.V4,
        features.V5,
        features.V6,
        features.V7,
        features.V8,
        features.V9,
        features.V10,
        features.V11,
        features.V12,
        features.V13,
        features.V14,
        features.V15,
        features.V16,
        features.V17,
        features.V18,
        features.V19,
        features.V20,
        features.V21,
        features.V22,
        features.V23,
        features.V24,
        features.V25,
        features.V26,
        features.V27,
        features.V28,
        features.Amount,
    ]

    # Convert to numpy array and reshape for single prediction
    X = np.array(feature_values).reshape(1, -1)

    # Get prediction and probability
    prediction = stacking_clf.predict(X)[0]
    probabilities = stacking_clf.predict_proba(X)[0]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probabilities[1]),
        "confidence": float(max(probabilities)),
    }
