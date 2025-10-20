from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# StandardScaler Wrapper for Pipeline Use
class DataStandardiser:
    """Standardize features using StandardScaler for pipeline use."""

    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on training data and transform."""
        X_scaled = self.scaler.fit_transform(X_train)
        self._is_fitted = True
        return pd.DataFrame(X_scaled, columns=X_train.columns, index=X_train.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform() first.")

        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def fit_transform_splits(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit on train, transform all splits."""
        X_train_scaled = self.fit_transform(X_train)
        X_val_scaled = self.transform(X_val)
        X_test_scaled = self.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled


def check_scaling_validity(X_train_scaled, X_val_scaled, X_test_scaled):
    """
    Plot distributions of scaled features to check validity of scaling.

    Parameters:
    -----------
    X_train_scaled : pd.DataFrame
        Scaled training feature set
    X_val_scaled : pd.DataFrame
        Scaled validation feature set
    X_test_scaled : pd.DataFrame
        Scaled test feature set
    """
    # Combine data for plotting
    data = pd.concat(
        [X_train_scaled, X_val_scaled, X_test_scaled], axis=0, keys=["Train", "Validation", "Test"]
    )

    print("Feature means after scaling:\n", data.mean())
    print("-----------------------------------")
    print("Feature std devs after scaling:\n", data.std())


class ApplySmote:
    """Apply SMOTE (Synthetic Minority Oversampling Technique) for class balancing."""
    
    def __init__(self, random_state=42, k_neighbors=5, sampling_strategy='auto'):
        """
        Initialize SMOTE with configurable parameters.
        
        Parameters:
        -----------
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
            sampling_strategy=sampling_strategy
        )
        self._is_fitted = False
    
    def fit_resample(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply SMOTE to balance the training data.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.DataFrame
            Training labels
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Balanced training features and labels
        """
        # Convert to numpy arrays for SMOTE, then back to DataFrames
        X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train.squeeze())
        self._is_fitted = True
        
        # Return as DataFrames with proper column names
        X_resampled_df = pd.DataFrame(X_resampled, columns=X_train.columns)
        y_resampled_df = pd.DataFrame(y_resampled, columns=y_train.columns)
        
        return X_resampled_df, y_resampled_df
    
    def print_class_distribution(self, y_before: pd.DataFrame, y_after: pd.DataFrame) -> None:
        """
        Print class distribution before and after SMOTE application.
        
        Parameters:
        -----------
        y_before : pd.DataFrame
            Labels before SMOTE
        y_after : pd.DataFrame  
            Labels after SMOTE
        """
        print('✅ Before SMOTE:')
        print(y_before.value_counts())
        print('-' * 25)
        print('✅ After SMOTE:')
        print(y_after.value_counts())


