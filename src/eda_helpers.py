# - Data Handling -
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class OutlierAnalyser:
    def __init__(self, palette="pastel"):
        self.palette = palette

    @staticmethod
    def inspect_outliers_iqr(df: pd.DataFrame):
        """Inspect outliers for all numeric columns using IQR method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = []

        for col_name in numeric_cols:
            try:
                # Calculate Q1, Q3, and IQR
                q1, q3 = df[col_name].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

                # Count outliers
                outlier_mask = (df[col_name] < lower) | (df[col_name] > upper)
                outlier_count = outlier_mask.sum()
                outlier_pct = round(outlier_count / len(df) * 100, 2)

                results.append(
                    {
                        "Column": col_name,
                        "Q1": f"{q1:.2f}",
                        "Q3": f"{q3:.2f}",
                        "Lower_Bound": f"{lower:.2f}",
                        "Upper_Bound": f"{upper:.2f}",
                        "Min": f"{df[col_name].min():.2f}",
                        "Max": f"{df[col_name].max():.2f}",
                        "Outlier_Count": outlier_count,
                        "Outlier_Pct": f"{outlier_pct}%",
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "Column": col_name,
                        "Q1": "Error",
                        "Q3": "Error",
                        "Lower_Bound": "Error",
                        "Upper_Bound": "Error",
                        "Min": "Error",
                        "Max": "Error",
                        "Outlier_Count": "Error",
                        "Outlier_Pct": f"Error: {str(e)}",
                    }
                )

        result_df = pd.DataFrame(results)
        result_df["Outlier_Pct_Numeric"] = (
            result_df["Outlier_Pct"]
            .str.replace("%", "")
            .str.replace("Error:.*", "0", regex=True)
            .astype(float)
        )
        result_df = (
            result_df.sort_values("Outlier_Pct_Numeric", ascending=False)
            .drop("Outlier_Pct_Numeric", axis=1)
            .reset_index(drop=True)
        )
        return result_df

    @staticmethod
    def violinplots_numeric_columns(
        df: pd.DataFrame, target_col: str = "Class", palette: str = "pastel", max_cols=3
    ):
        """Generate violinplots for numerical columns to detect outliers and skewness."""
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        n_cols = max_cols
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        fig.suptitle("Violin Plots of Numeric Features", y=1, size=25)
        axs = axs.flatten() if n_rows > 1 else [axs]
        
        for i, col in enumerate(numeric_cols):
            sns.violinplot(data=df, y=col, hue=target_col,  ax=axs[i], palette=palette, inner_kws=dict(box_width=15, whis_width=2, color=".8"))
            axs[i].set_title(col + ', skewness is: '+str(round(df[col].skew(axis = 0, skipna = True),2)))


        # Remove unused subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()
        
