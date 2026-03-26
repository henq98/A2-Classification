import numpy as np
import matplotlib.pyplot as plt


FEATURE_NAMES = [
    "height",
    "root_density",
    "area",
    "shape_index",
    "linearity",
    "sphericity",
    "slenderness",
    "length_height_ratio",
    "circularity",
    "footprint_density",
]

CLASS_LABELS = ["building", "car", "fence", "pole", "tree"]
CLASS_COLORS = ["firebrick", "grey", "darkorange", "dodgerblue", "olivedrab"]


def scatter_two_features(X, y, feat_x, feat_y, log_y=False, log_x=False):
    fig, ax = plt.subplots(figsize=(9, 6))

    xvals = X[:, feat_x].copy()
    yvals = X[:, feat_y].copy()

    if log_x:
        xvals = np.log1p(np.maximum(xvals, 0))
    if log_y:
        yvals = np.log1p(np.maximum(yvals, 0))

    for cls in range(5):
        mask = y == cls
        ax.scatter(
            xvals[mask],
            yvals[mask],
            s=70,
            c=CLASS_COLORS[cls],
            edgecolor="k",
            label=CLASS_LABELS[cls],
            alpha=0.8,
        )

    xlabel = FEATURE_NAMES[feat_x]
    ylabel = FEATURE_NAMES[feat_y]

    if log_x:
        xlabel = f"log(1 + {xlabel})"
    if log_y:
        ylabel = f"log(1 + {ylabel})"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{xlabel} vs {ylabel}")
    ax.legend()
    plt.tight_layout()
    plt.show()


def scatter_two_features(X, y, feat_x, feat_y, log_y=False, log_x=False):
    fig, ax = plt.subplots(figsize=(9, 6))

    xvals = X[:, feat_x].copy()
    yvals = X[:, feat_y].copy()

    if log_x:
        xvals = np.log1p(np.maximum(xvals, 0))
    if log_y:
        yvals = np.log1p(np.maximum(yvals, 0))

    for cls in range(5):
        mask = y == cls
        ax.scatter(
            xvals[mask],
            yvals[mask],
            s=70,
            c=CLASS_COLORS[cls],
            edgecolor="k",
            label=CLASS_LABELS[cls],
            alpha=0.8
        )

    xlabel = FEATURE_NAMES[feat_x]
    ylabel = FEATURE_NAMES[feat_y]

    if log_x:
        xlabel = f"log(1 + {xlabel})"
    if log_y:
        ylabel = f"log(1 + {ylabel})"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{xlabel} vs {ylabel}")
    ax.legend()
    plt.tight_layout()
    plt.show()