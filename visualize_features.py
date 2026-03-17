import numpy as np
import matplotlib.pyplot as plt


FEATURE_NAMES = [
    "height", "mean_z", "std_z", "root_density", "area", "shape_index", "circularity",
    "elongation_xy", "slenderness", "rectangularity", "footprint_density",
    "bbox_volume", "point_density_3d", "lower_fraction", "middle_fraction",
    "upper_fraction", "top_roughness", "linearity", "planarity",
    "sphericity", "anisotropy", "curvature"
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