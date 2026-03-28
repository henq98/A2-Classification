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


def load_data(data_file="data.txt"):
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=",", comments="#")
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)
    return ID, X, y


def scatter_two_features(X, y, feat_x, feat_y, log_y=False, log_x=False, save_path=None, show=True):
    """
    Scatter plot of two selected features for the 5 classes.
    """
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
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()


def boxplot_feature(X, y, feat_idx, save_path=None, show=True):
    """
    Boxplot of one feature per class.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    data_by_class = []
    for cls in range(5):
        data_by_class.append(X[y == cls, feat_idx])

    ax.boxplot(data_by_class, tick_labels=CLASS_LABELS)
    ax.set_title(f"Boxplot of {FEATURE_NAMES[feat_idx]}")
    ax.set_ylabel(FEATURE_NAMES[feat_idx])
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    ID, X, y = load_data("data.txt")

    scatter_two_features(X, y, feat_x=6, feat_y=7)               # slenderness vs length_height_ratio
    scatter_two_features(X, y, feat_x=8, feat_y=9)               # circularity vs footprint_density
    scatter_two_features(X, y, feat_x=1, feat_y=2, log_y=True)   # root_density vs area

    boxplot_feature(X, y, feat_idx=0)   # height
    boxplot_feature(X, y, feat_idx=6)   # slenderness
    boxplot_feature(X, y, feat_idx=8)   # circularity