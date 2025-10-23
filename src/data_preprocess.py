from typing import Tuple

from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split



# - Class for Data Cleaning Tasks -
class DataCleaner:
    # def remove_correlated_features(df, threshold=0.65):
    # def remove_outliers(df, z_thresh=3):

    @staticmethod
    def drop_duplicates(df):
        """
        Remove duplicate rows from the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            Dataframe with duplicates removed
        """
        initial_shape = df.shape
        df = df.drop_duplicates()
        final_shape = df.shape
        print(f"✅ Initial Shape (Before dropping duplicates): {initial_shape}")
        print(f"✅ Dropped {initial_shape[0] - final_shape[0]} duplicate rows.")
        print(f"✅ Final Shape (After dropping duplicates): {final_shape}")
        return df


# - StandardScaler Validity Check -
def check_scaling_validity(X_df_scaled: pd.DataFrame):
    """
    Plot distributions of scaled features to check validity of scaling.

    Parameters
    ----------
    X_df_scaled : pd.DataFrame
        Scaled feature set to undergo checking
    """
    # Combine data for plotting

    print("✅ Feature means after scaling:\n", X_df_scaled.mean())
    print("-----------------------------------")
    print("\n✅ Feature std devs after scaling:\n", X_df_scaled.std())


# - Train-Val-Test Splitter -
def train_val_test_split(
    df: pd.DataFrame, target_column: str = "Class"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split dataframe into train-validation-test sets with stratification (60% train, 20% val, 20% test).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and target
    target_column : str, default='Class'
        Name of the target column

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
        X_train, X_val, X_test, y_train, y_val, y_test
    """

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # 1. Train-Val and Test Split (80% train_val, 20% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
    )

    # 2. Train and Validation Split (75% train, 25% val of train_val)
    # This results in: 60% train, 20% val, 20% test
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, shuffle=True, random_state=42, stratify=y_train_val
    )

    # Print split information
    print("Initial Dataset shape:", X.shape)
    print("-----------------------------")
    print(
        "X_train shape:",
        X_train.shape,
        "Pct:",
        f"{round(X_train.shape[0] / X.shape[0] * 100, 2)}%",
    )
    print("y_train shape:", y_train.shape)
    print("-----------------------------")
    print("X_val shape:", X_val.shape, "Pct:", f"{round(X_val.shape[0] / X.shape[0] * 100, 2)}%")
    print("y_val shape:", y_val.shape)
    print("-----------------------------")
    print(
        "X_test shape:", X_test.shape, "Pct:", f"{round(X_test.shape[0] / X.shape[0] * 100, 2)}%"
    )
    print("y_test shape:", y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test



#! Legacy SMOTE Applier (DELETE SOON & WRITE PLOTTING CODE SEPARATELY)
class ApplySmote:
    """Apply SMOTE (Synthetic Minority Oversampling Technique) for class balancing."""

    def __init__(self, random_state=42, k_neighbors=5):
        """
        Initialize SMOTE with configurable parameters.

        Parameters
        ----------
        random_state : int, default=42
            Random state for reproducibility
        k_neighbors : int, default=5
            Number of nearest neighbors for SMOTE algorithm
        sampling_strategy : str or dict, default='auto'
            Sampling strategy for class balancing
        """
        self.smote = SMOTE(
            random_state=random_state,
            k_neighbors=k_neighbors,
        )
        self._is_fitted = False

    @staticmethod
    def fit_resample(
        X_train: pd.DataFrame, y_train: pd.Series, random_state=42, k_neighbors=5
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to balance the training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        random_state : int, default=42
            Random state for reproducibility
        k_neighbors : int, default=5
            Number of nearest neighbors for SMOTE algorithm

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Balanced training features and labels
        """
        # Initialize SMOTE with parameters
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)

        # Convert to numpy arrays for SMOTE, then back to DataFrames
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # Return as DataFrames with proper column names
        X_resampled_df = pd.DataFrame(X_resampled, columns=X_train.columns)
        y_resampled_series = pd.Series(y_resampled, name="Class")

        # Print class distribution comparison
        ApplySmote._plot_class_distribution(y_train, y_resampled_series)

        return X_resampled_df, y_resampled_series

    @staticmethod
    def _plot_class_distribution(y_before: pd.Series, y_after: pd.Series) -> None:
        """
        Plot class distribution before and after SMOTE application with annotations.

        Parameters
        ----------
        y_before : pd.Series
            Labels before SMOTE
        y_after : pd.Series
            Labels after SMOTE
        """
        # Plot class distribution after over-sampling
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Before SMOTE plot
        counts_before = y_before.value_counts().sort_index()
        total_before = len(y_before)
        counts_before.plot(kind="bar", color=["#A3C1E0", "#F4C3D7"], ax=axes[0])
        axes[0].set_title("Class Distribution Before Over-Sampling (SMOTE)")
        axes[0].set_xlabel("Class (0: Non-Fraud, 1: Fraud)")
        axes[0].set_ylabel("Number of Transactions")
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(["Non-Fraud", "Fraud"], rotation=0)

        # Add annotations for before plot
        for i, (class_label, count) in enumerate(counts_before.items()):
            percentage = (count / total_before) * 100
            axes[0].annotate(
                f"{count:,} ({percentage:.1f}%)",
                (i, count),
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # After SMOTE plot
        counts_after = y_after.value_counts().sort_index()
        total_after = len(y_after)
        counts_after.plot(kind="bar", color=["#A3C1E0", "#F4C3D7"], ax=axes[1])
        axes[1].set_title("Class Distribution After Over-Sampling (SMOTE)")
        axes[1].set_xlabel("Class (0: Non-Fraud, 1: Fraud)")
        axes[1].set_ylabel("Number of Transactions")
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["Non-Fraud", "Fraud"], rotation=0)

        # Add annotations for after plot
        for i, (class_label, count) in enumerate(counts_after.items()):
            percentage = (count / total_after) * 100
            axes[1].annotate(
                f"{count:} ({percentage:.1f}%)",
                (i, count),
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        plt.tight_layout()
        plt.show()


class SmoteApplier(BaseEstimator):
    def __init__(self, random_state=42, k_neighbors=5):
        # Store parameters as instance attributes for sklearn compatibility
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)

    def fit(self, X, y=None):
        # No fitting just return self, SMOTE is done at fit_resample stage
        return self
    
    def fit_resample(self, X, y):
        """Method called by imblearn Pipeline for resampling"""
        X_res, y_res = self.smote.fit_resample(X, y)
        X_resampled_df = pd.DataFrame(X_res, columns=X.columns)
        y_resampled_series = pd.Series(y_res, name="Class")
        self._plot_class_distribution(y, y_res)
        return X_resampled_df, y_resampled_series

    @staticmethod
    def _plot_class_distribution(y_before, y_after):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        counts_before = y_before.value_counts().sort_index()
        counts_after = y_after.value_counts().sort_index()
        total_before = len(y_before)
        total_after = len(y_after)

        counts_before.plot(kind="bar", ax=axes[0], color=["#A3C1E0", "#F4C3D7"])
        axes[0].set(title="Before SMOTE", xlabel="Class", ylabel="Count")
        axes[0].set_xticklabels(["Non-Fraud", "Fraud"], rotation=0)
        for i, count in enumerate(counts_before):
            axes[0].annotate(f"{count} ({count/total_before*100:.1f}%)", (i, count),
                             ha="center", va="bottom", fontweight="bold")

        counts_after.plot(kind="bar", ax=axes[1], color=["#A3C1E0", "#F4C3D7"])
        axes[1].set(title="After SMOTE", xlabel="Class", ylabel="Count")
        axes[1].set_xticklabels(["Non-Fraud", "Fraud"], rotation=0)
        for i, count in enumerate(counts_after):
            axes[1].annotate(f"{count} ({count/total_after*100:.1f}%)", (i, count),
                             ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        plt.show()