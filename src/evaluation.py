from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import src.plots


class ResultsAnalyser:
    """Analyze and visualize model performance results."""

    def __init__(self, model_name="Model"):
        self.model_name = model_name

    def analyze_predictions(
        self, y_true, y_pred, y_pred_proba=None, class_labels=["Non-Fraud", "Fraud"]
    ):
        """
        Complete analysis of model predictions.

        Parameters:
        -----------
        y_true : array-like
            True class labels
        y_pred : array-like
            Predicted class labels
        y_pred_proba : array-like, optional
            Prediction probabilities for positive class
        class_labels : list
            Class label names for confusion matrix (Majority Class 1st)
        """
        print(f"✅ {self.model_name} Results Analysis")
        print("=" * 50)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Print key metrics
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        # AUC metrics if probabilities provided
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
                pr_auc = average_precision_score(y_true, y_pred_proba)
                print(f"ROC-AUC:   {roc_auc:.4f}")
                print(f"PR-AUC:    {pr_auc:.4f}")
            except ValueError:
                print("ROC-AUC:   N/A (only one class present)")
                print("PR-AUC:    N/A (only one class present)")

        print("\n✅ Classification Report:")
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        print(report)
        print("-" * 20)
        # Plot confusion matrix
        print("\n✅ Confusion Matrix:")
        if class_labels is None:
            class_labels = ["Class 0", "Class 1"]
        src.plots.plot_confusion_matrix(y_true, y_pred, class_labels)

        # Return metrics dictionary
        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

        if y_pred_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
                metrics["pr_auc"] = average_precision_score(y_true, y_pred_proba)
            except ValueError:
                metrics["roc_auc"] = None
                metrics["pr_auc"] = None

        return metrics
