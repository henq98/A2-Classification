import numpy as np
import matplotlib.pyplot as plt


FEATURE_NAMES = [
    "height",
    "root_density",
    "area",
    "shape_index",
    "linearity",
    "sphericity",
]

CLASS_LABELS = ["building", "car", "fence", "pole", "tree"]
CLASS_COLORS = ["firebrick", "grey", "darkorange", "dodgerblue", "olivedrab"]


def load_data(data_file="data.txt"):
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=",", comments="#")
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)
    return ID, X, y


def scatter_two_features(X, y, feat_x, feat_y, log_y=False, log_x=False):
    fig, ax = plt.subplots(figsize=(9, 6))

    xvals = X[:, feat_x].copy()
    yvals = X[:, feat_y].copy()

    if log_x:
        xvals = np.log1p(xvals)
    if log_y:
        yvals = np.log1p(yvals)

    for cls in range(5):
        mask = y == cls
        ax.scatter(
            xvals[mask],
            yvals[mask],
            s=90,
            c=CLASS_COLORS[cls],
            edgecolor="k",
            label=CLASS_LABELS[cls],
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


def scatter_matrix_style(X, y):
    pairs = [
        (1, 2),  # root_density vs area
        (0, 2),  # height vs area
        (3, 4),  # shape_index vs linearity
        (4, 5),  # linearity vs sphericity
    ]

    for fx, fy in pairs:
        scatter_two_features(X, y, fx, fy)


def boxplot_feature(X, y, feat_idx):
    grouped = [X[y == cls, feat_idx] for cls in range(5)]

    plt.figure(figsize=(9, 6))
    plt.boxplot(grouped, tick_labels=CLASS_LABELS)
    plt.ylabel(FEATURE_NAMES[feat_idx])
    plt.title(f"Boxplot of {FEATURE_NAMES[feat_idx]} per class")
    plt.tight_layout()
    plt.show()


def histogram_feature(X, y, feat_idx, bins=20, log_x=False):
    plt.figure(figsize=(9, 6))

    for cls in range(5):
        vals = X[y == cls, feat_idx]
        if log_x:
            vals = np.log1p(vals)
        plt.hist(vals, bins=bins, alpha=0.5, label=CLASS_LABELS[cls])

    xlabel = FEATURE_NAMES[feat_idx]
    if log_x:
        xlabel = f"log(1 + {xlabel})"

    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(f"Histogram of {xlabel}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ID, X, y = load_data("data.txt")

    # original idea: root_density vs area
    scatter_two_features(X, y, feat_x=1, feat_y=2)

    # better for skewed area values
    scatter_two_features(X, y, feat_x=1, feat_y=2, log_y=True)

    # try a few useful pairs
    scatter_matrix_style(X, y)

    # inspect single-feature distributions
    boxplot_feature(X, y, feat_idx=2)          # area
    histogram_feature(X, y, feat_idx=2, log_x=True)