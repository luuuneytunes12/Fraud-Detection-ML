import matplotlib.pyplot as plt
import mlflow
import numpy as np
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


"""
MLflow K-Fold CV Evaluation - Class-Based Module
Logs all metrics, plots (as images), and classification reports to MLflow.
"""

from sklearn.base import clone
from sklearn.metrics import (
    auc,
    fbeta_score,
    log_loss,
    make_scorer,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve

# Optional imports - handle gracefully if not available
try:
    from sklearn.metrics import calibration_curve
except ImportError:
    calibration_curve = None

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from scipy.special import expit
except ImportError:
    try:
        from sklearn.utils.fixes import expit
    except ImportError:

        def expit(x):
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))


class MLflowExperimentRun:
    """
    Super simple MLflow experiment runner for sklearn models/pipelines.

    Perfect for MLflow and sklearn beginners - just pass your model and data!

    Example Usage:
    -------------
    from lightgbm import LGBMClassifier
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE

    # Create your model/pipeline
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('lgbm', LGBMClassifier(random_state=42))
    ])

    # Run experiment (that's it!)
    runner = MLflowExperimentRun(
        model=pipeline,
        experiment_name="Fraud Detection",
        run_name="LightGBM_SMOTE"
    )

    # Train + Cross-validation + Test evaluation
    runner.run_experiment(X_train, y_train, X_test, y_test)

    # Or just train + cross-validation
    runner.run_experiment(X_train, y_train)
    """

    def __init__(
        self,
        model,
        experiment_name="ML Experiment",
        run_name="Model Run",
        cv_folds=5,
        random_state=42,
    ):
        """
        Initialize MLflow experiment runner.

        Parameters
        ----------
        model : sklearn-compatible model or pipeline
            Any sklearn model or imblearn pipeline
        experiment_name : str, default="ML Experiment"
            Name of MLflow experiment
        run_name : str, default="Model Run"
            Name for the main run
        cv_folds : int, default=5
            Number of cross-validation folds
        random_state : int, default=42
            Random state for reproducibility
        """
        self.model = model
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        self.f2_scorer = make_scorer(fbeta_score, beta=2)

        # Set up MLflow experiment
        self._setup_experiment()

    def _setup_experiment(self):
        """Set up or get MLflow experiment."""
        try:
            exp = mlflow.get_experiment_by_name(self.experiment_name)
            if exp is None:
                mlflow.create_experiment(self.experiment_name)
                print(f"✅ Created experiment: {self.experiment_name}")
            mlflow.set_experiment(self.experiment_name)
            print(f"✅ Using experiment: {self.experiment_name}")
        except Exception as e:
            print(f"⚠️ Experiment setup warning: {e}")

    def run_experiment(self, X_train, y_train, X_test=None, y_test=None):
        """
        Run complete MLflow experiment with train/CV/test evaluation.

        Parameters
        ----------
        X_train : array-like or DataFrame
            Training features
        y_train : array-like or Series
            Training labels
        X_test : array-like or DataFrame, optional
            Test features (if None, only train+CV evaluation)
        y_test : array-like or Series, optional
            Test labels (if None, only train+CV evaluation)

        Returns
        -------
        dict
            Summary of all logged metrics
        """
        print(f"🚀 Starting MLflow experiment: {self.experiment_name}")
        print(f"📊 Model: {type(self.model).__name__}")
        print(f"🔄 Cross-validation: {self.cv_folds} folds")
        print(f"🧪 Test evaluation: {'Yes' if X_test is not None else 'No'}")
        print("=" * 60)

        all_metrics = {}

        # Disable auto-logging to have full control
        mlflow.sklearn.autolog(disable=True)
        mlflow.lightgbm.autolog(disable=True)

        with mlflow.start_run(run_name=self.run_name) as parent_run:
            print(f"📝 Started parent run: {self.run_name}")

            # Log basic experiment info
            mlflow.log_params(
                {
                    "model_type": type(self.model).__name__,
                    "cv_folds": self.cv_folds,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test) if X_test is not None else 0,
                    "random_state": self.random_state,
                }
            )

            # 1. TRAIN SET EVALUATION
            print("\n🎯 Training model on full dataset...")
            train_metrics = self._evaluate_train_set(X_train, y_train, parent_run.info.run_id)
            all_metrics.update(train_metrics)

            # 2. CROSS-VALIDATION EVALUATION
            print("\n🔄 Running cross-validation...")
            cv_metrics = self._evaluate_cross_validation(X_train, y_train, parent_run.info.run_id)
            all_metrics.update(cv_metrics)

            # 3. TEST SET EVALUATION (if provided)
            if X_test is not None and y_test is not None:
                print("\n🧪 Evaluating on test set...")
                test_metrics = self._evaluate_test_set(
                    X_train, y_train, X_test, y_test, parent_run.info.run_id
                )
                all_metrics.update(test_metrics)

            # 4. LOG FINAL MODEL
            print("\n💾 Logging final model...")
            try:
                mlflow.sklearn.log_model(
                    self.model,
                    artifact_path="final_model",
                    input_example=X_train[:5] if hasattr(X_train, "iloc") else X_train[:5],
                )
            except Exception as e:
                print(f"⚠️ Model logging warning: {e}")

            print(f"\n✅ Experiment complete! Run ID: {parent_run.info.run_id}")
            print("=" * 60)

        return all_metrics

    def _evaluate_train_set(self, X_train, y_train, parent_run_id):
        """Evaluate model on training set and log to MLflow."""
        with mlflow.start_run(run_name="train_evaluation", nested=True) as train_run:
            print(f"   📝 Train evaluation run: {train_run.info.run_id}")

            # Fit model
            self.model.fit(X_train, y_train)

            # Get predictions
            y_pred = self.model.predict(X_train)
            y_pred_proba = self._get_prediction_probabilities(X_train)

            # Calculate and log metrics
            metrics = self._calculate_all_metrics(y_train, y_pred, y_pred_proba, "train")
            mlflow.log_metrics(metrics)

            # Create and log plots
            self._log_all_plots(y_train, y_pred, y_pred_proba, "train", X_train)

            return metrics

    def _evaluate_cross_validation(self, X_train, y_train, parent_run_id):
        """Run cross-validation with child runs for each fold."""
        with mlflow.start_run(run_name="cross_validation", nested=True) as cv_parent_run:
            print(f"   📝 CV parent run: {cv_parent_run.info.run_id}")

            # Get CV predictions
            y_pred_cv = cross_val_predict(
                self.model, X_train, y_train, cv=self.cv, method="predict"
            )
            y_pred_proba_cv = self._get_cv_prediction_probabilities(X_train, y_train)

            # Calculate overall CV metrics
            cv_metrics = self._calculate_all_metrics(y_train, y_pred_cv, y_pred_proba_cv, "cv")
            mlflow.log_metrics(cv_metrics)

            # Create and log CV plots
            self._log_all_plots(y_train, y_pred_cv, y_pred_proba_cv, "cv", X_train)

            # Run individual fold evaluations as child runs
            self._evaluate_individual_folds(X_train, y_train, cv_parent_run.info.run_id)

            return cv_metrics

    def _evaluate_individual_folds(self, X_train, y_train, cv_parent_run_id):
        """Evaluate each CV fold as separate child runs."""
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X_train, y_train)):
            with mlflow.start_run(run_name=f"fold_{fold_idx + 1}", nested=True):
                print(f"     📂 Fold {fold_idx + 1}")

                # Split data for this fold
                if hasattr(X_train, "iloc"):  # DataFrame
                    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                else:  # numpy array
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                # Train model on fold
                fold_model = self._clone_model()
                fold_model.fit(X_fold_train, y_fold_train)

                # Get predictions
                y_pred_fold = fold_model.predict(X_fold_val)
                y_pred_proba_fold = self._get_prediction_probabilities(X_fold_val, fold_model)

                # Calculate and log metrics
                fold_metrics = self._calculate_all_metrics(
                    y_fold_val, y_pred_fold, y_pred_proba_fold, f"fold_{fold_idx + 1}"
                )
                mlflow.log_metrics(fold_metrics)

                # Log fold-specific info
                mlflow.log_params(
                    {
                        "fold_number": fold_idx + 1,
                        "train_samples": len(X_fold_train),
                        "val_samples": len(X_fold_val),
                    }
                )

    def _evaluate_test_set(self, X_train, y_train, X_test, y_test, parent_run_id):
        """Evaluate on test set and log to MLflow."""
        with mlflow.start_run(run_name="test_evaluation", nested=True) as test_run:
            print(f"   📝 Test evaluation run: {test_run.info.run_id}")

            # Use already fitted model from train evaluation
            y_pred_test = self.model.predict(X_test)
            y_pred_proba_test = self._get_prediction_probabilities(X_test)

            # Calculate and log metrics
            test_metrics = self._calculate_all_metrics(
                y_test, y_pred_test, y_pred_proba_test, "test"
            )
            mlflow.log_metrics(test_metrics)

            # Create and log plots
            self._log_all_plots(y_test, y_pred_test, y_pred_proba_test, "test", X_test)

            return test_metrics

    def _calculate_all_metrics(self, y_true, y_pred, y_pred_proba, prefix):
        """Calculate all metrics for logging."""
        metrics = {}

        # Basic metrics
        metrics[f"{prefix}_accuracy"] = accuracy_score(y_true, y_pred)
        metrics[f"{prefix}_precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f"{prefix}_recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics[f"{prefix}_f1"] = f1_score(y_true, y_pred, zero_division=0)
        metrics[f"{prefix}_f2"] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

        # AUC metrics (if probabilities available)
        if y_pred_proba is not None:
            try:
                metrics[f"{prefix}_roc_auc"] = roc_auc_score(y_true, y_pred_proba)
                metrics[f"{prefix}_pr_auc"] = average_precision_score(y_true, y_pred_proba)
                metrics[f"{prefix}_log_loss"] = log_loss(y_true, y_pred_proba)
            except ValueError:
                # Handle cases with only one class
                metrics[f"{prefix}_roc_auc"] = 0.0
                metrics[f"{prefix}_pr_auc"] = 0.0
                metrics[f"{prefix}_log_loss"] = 0.0

        return metrics

    def _log_all_plots(self, y_true, y_pred, y_pred_proba, prefix, X_data):
        """Create and log all plots to MLflow."""
        try:
            # 1. Confusion Matrix
            self._log_confusion_matrix(y_true, y_pred, prefix)

            # 2. Classification Report (as text artifact)
            self._log_classification_report(y_true, y_pred, prefix)

            if y_pred_proba is not None:
                # 3. ROC Curve
                self._log_roc_curve(y_true, y_pred_proba, prefix)

                # 4. Precision-Recall Curve
                self._log_pr_curve(y_true, y_pred_proba, prefix)

                # 5. Calibration Plot
                self._log_calibration_plot(y_true, y_pred_proba, prefix)

            # 6. Learning Curve (only for train/cv, not individual folds)
            if prefix in ["train", "cv"]:
                self._log_learning_curve(X_data, y_true, prefix)

            # 7. Feature Importance (SHAP) - only for train to avoid redundancy
            if prefix == "train":
                self._log_shap_feature_importance(X_data)

        except Exception as e:
            print(f"⚠️ Warning: Could not create some plots: {e}")

    def _log_confusion_matrix(self, y_true, y_pred, prefix):
        """Log confusion matrix plot."""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-Fraud", "Fraud"],
            yticklabels=["Non-Fraud", "Fraud"],
        )
        plt.title(f"Confusion Matrix ({prefix.upper()})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), f"{prefix}_confusion_matrix.png")
        plt.close()

    def _log_classification_report(self, y_true, y_pred, prefix):
        """Log classification report as text artifact."""
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        mlflow.log_text(report, f"{prefix}_classification_report.txt")

    def _log_roc_curve(self, y_true, y_pred_proba, prefix):
        """Log ROC curve plot."""
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({prefix.upper()})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), f"{prefix}_roc_curve.png")
        plt.close()

    def _log_pr_curve(self, y_true, y_pred_proba, prefix):
        """Log Precision-Recall curve plot."""
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.plot(recall, precision, label="PR Curve", linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve ({prefix.upper()})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), f"{prefix}_pr_curve.png")
        plt.close()

    def _log_calibration_plot(self, y_true, y_pred_proba, prefix):
        """Log calibration plot."""
        if calibration_curve is None:
            print("⚠️ Calibration plot not available (sklearn version issue)")
            return

        try:
            plt.figure(figsize=(8, 6))
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title(f"Calibration Plot ({prefix.upper()})")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), f"{prefix}_calibration_plot.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not create calibration plot: {e}")

    def _log_learning_curve(self, X, y, prefix):
        """Log learning curve plot."""
        try:
            plt.figure(figsize=(10, 6))
            learning_result = learning_curve(
                self.model,
                X,
                y,
                cv=self.cv,
                scoring=self.f2_scorer,
                train_sizes=np.linspace(0.1, 1.0, 5),
                n_jobs=-1,
            )

            # Handle different sklearn versions that return different numbers of values
            if len(learning_result) == 3:
                train_sizes, train_scores, val_scores = learning_result
            elif len(learning_result) == 5:
                train_sizes, train_scores, val_scores, _, _ = learning_result
            else:
                train_sizes, train_scores, val_scores = learning_result[:3]

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            plt.plot(train_sizes, train_mean, "o-", label="Training F2 Score")
            plt.fill_between(
                train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1
            )
            plt.plot(train_sizes, val_mean, "o-", label="Validation F2 Score")
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)

            plt.xlabel("Training Set Size")
            plt.ylabel("F2 Score")
            plt.title(f"Learning Curve ({prefix.upper()})")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), f"{prefix}_learning_curve.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not create learning curve: {e}")

    def _log_shap_feature_importance(self, X):
        """Log SHAP feature importance plot."""
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP not available - install with: pip install shap")
            return

        try:
            # Create a small sample for SHAP (it can be slow on large datasets)
            sample_size = min(100, len(X))
            if hasattr(X, "sample"):  # DataFrame
                X_sample = X.sample(n=sample_size, random_state=self.random_state)
            else:  # numpy array
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[indices]

            # Create SHAP explainer
            explainer = shap.Explainer(self.model.predict, X_sample)
            shap_values = explainer(X_sample)

            # Create summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "feature_importance_shap.png")
            plt.close()

        except Exception as e:
            print(f"⚠️ Could not create SHAP plot: {e}")

    def _get_prediction_probabilities(self, X, model=None):
        """Get prediction probabilities, handling models that might not have predict_proba."""
        try:
            if model is None:
                model = self.model
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                # Convert decision function to probabilities
                return expit(model.decision_function(X))
            else:
                return None
        except Exception:
            return None

    def _get_cv_prediction_probabilities(self, X, y):
        """Get cross-validation prediction probabilities."""
        try:
            return cross_val_predict(self.model, X, y, cv=self.cv, method="predict_proba")[:, 1]
        except Exception:
            try:
                return cross_val_predict(self.model, X, y, cv=self.cv, method="decision_function")
            except Exception:
                return None

    def _clone_model(self):
        """Clone the model for individual fold training."""
        return clone(self.model)
