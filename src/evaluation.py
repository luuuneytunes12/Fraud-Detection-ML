import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ResultsAnalyser:
    """Analyze and visualize model performance results."""

    @staticmethod
    def analyze_predictions(
        y_true, y_pred, y_pred_proba=None, class_labels=["Non-Fraud", "Fraud"], model_name="Model"
    ):
        """
        Complete analysis of model predictions.

        Parameters
        ----------
        y_true : array-like
            True class labels
        y_pred : array-like
            Predicted class labels
        y_pred_proba : array-like, optional
            Prediction probabilities for positive class
        class_labels : list, default=["Non-Fraud", "Fraud"]
            Class label names for confusion matrix (Majority Class 1st)
        model_name : str, default="Model"
            Name of the model for display purposes

        Returns
        -------
        dict
            Dictionary containing calculated metrics
        """
        print(f"✅ {model_name} Results Analysis")
        print("=" * 50)

        # Calculate basic metrics
        metrics = ResultsAnalyser._calculate_basic_metrics(y_true, y_pred)

        # Print basic metrics
        ResultsAnalyser._print_basic_metrics(metrics)

        # Calculate and print AUC metrics if probabilities provided
        if y_pred_proba is not None:
            auc_metrics = ResultsAnalyser._calculate_auc_metrics(y_true, y_pred_proba)
            metrics.update(auc_metrics)
            ResultsAnalyser._print_auc_metrics(auc_metrics)

        # Print classification report
        ResultsAnalyser._print_classification_report(y_true, y_pred)

        # Plot confusion matrix
        ResultsAnalyser._plot_confusion_matrix(y_true, y_pred, class_labels)

        return metrics

    @staticmethod
    def _calculate_basic_metrics(y_true, y_pred):
        """Calculate basic classification metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

    @staticmethod
    def _calculate_auc_metrics(y_true, y_pred_proba):
        """Calculate AUC-based metrics."""
        try:
            return {
                "roc_auc": roc_auc_score(y_true, y_pred_proba),
                "pr_auc": average_precision_score(y_true, y_pred_proba),
            }
        except ValueError:
            return {"roc_auc": None, "pr_auc": None}

    @staticmethod
    def _print_basic_metrics(metrics):
        """Print basic classification metrics."""
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")

    @staticmethod
    def _print_auc_metrics(auc_metrics):
        """Print AUC-based metrics."""
        if auc_metrics["roc_auc"] is not None:
            print(f"ROC-AUC:   {auc_metrics['roc_auc']:.4f}")
            print(f"PR-AUC:    {auc_metrics['pr_auc']:.4f}")
        else:
            print("ROC-AUC:   N/A (only one class present)")
            print("PR-AUC:    N/A (only one class present)")

    @staticmethod
    def _print_classification_report(y_true, y_pred):
        """Print detailed classification report."""
        print("\n✅ Classification Report:")
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        print(report)
        print("-" * 20)

    @staticmethod
    def _plot_confusion_matrix(y_true, y_pred, class_labels):
        """Plot confusion matrix."""
        print("\n✅ Confusion Matrix:")
        if class_labels is None:
            class_labels = ["Class 0", "Class 1"]

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create figure and plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
