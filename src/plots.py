# ==========================================
# Setting rcParams for Matplotlib
# ==========================================
from pathlib import Path

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt

# - Data Visualisation -
import pandas as pd
import seaborn as sns

# - ML Libraries -
from sklearn.metrics import confusion_matrix

# Define custom colour cycle using tab10 palette
colors = cycler(color=plt.get_cmap("tab10").colors)  # ['#1f77b4', '#ff7f0e', '#2ca02c', ...]

# Set overall style and appearance
mpl.style.use("ggplot")  # Use ggplot-style background and grid

# Global figure settings
mpl.rcParams["figure.figsize"] = (20, 5)  # Default figure size (width=20, height=5)
mpl.rcParams["figure.dpi"] = 100  # Resolution (dots per inch)
mpl.rcParams["figure.titlesize"] = 25  # Default title font size

# Axes and grid styling
mpl.rcParams["axes.facecolor"] = "white"  # Plot background colour
mpl.rcParams["axes.grid"] = True  # Enable grid
mpl.rcParams["grid.color"] = "lightgray"  # Grid line colour
mpl.rcParams["axes.prop_cycle"] = colors  # Set colour cycle for plots
mpl.rcParams["axes.linewidth"] = 1  # Border thickness

# Axis tick and label colours
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"

# Font settings
mpl.rcParams["font.size"] = 12  # Base font size for labels, ticks, etc.

# ==========================================
# Plot Functions
# ==========================================


def plot_class_distribution(data_dir="data/processed", subfolder=None):
    """
    Plot class distribution across train/validation/test splits.

    Parameters:
    -----------
    data_dir : str, default "data/processed"
        Base directory containing the data files
    subfolder : str, optional
        Additional subfolder path within data_dir
    """
    # Construct data path
    if subfolder:
        data_path = Path(data_dir) / subfolder
    else:
        data_path = Path(data_dir)

    # Load data
    try:
        y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
        y_val = pd.read_csv(data_path / "y_val.csv").squeeze()
        y_test = pd.read_csv(data_path / "y_test.csv").squeeze()
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Data for plotting
    splits = {"Train": y_train, "Validation": y_val, "Test": y_test}

    # Plot each split
    for ax, (name, y_split) in zip(axes, splits.items()):
        counts = y_split.value_counts().sort_index()
        total = len(y_split)

        # Create bar plot
        ax.bar(counts.index, counts.values, color=["skyblue", "salmon"])
        ax.set_title(f"{name} Set")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")

        # Add count and percentage labels
        for i, (class_val, count) in enumerate(counts.items()):
            pct = 100 * count / total
            ax.text(
                i,
                count + max(counts) * 0.01,
                f"{count:,} ({pct:.2f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_labels=None):
    """
    Plot confusion matrix for model predictions.

    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    class_labels : list, optional
        List of class label names for axis labels
    """
    # Create a Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Set up labels
    if class_labels is None:
        class_labels = ["Class 0", "Class 1"]

    # Create heatmap with custom labels
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
