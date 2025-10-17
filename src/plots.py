# ==========================================
# Setting rcParams for Matplotlib
# ==========================================
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt

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
