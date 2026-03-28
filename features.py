from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from sklearn.neighbors import KDTree
from tqdm import tqdm


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


class UrbanObject:
    def __init__(self, filename: str | Path):
        filename = Path(filename)
        self.cloud_name = filename.stem
        self.cloud_id = int(self.cloud_name)
        self.label = math.floor(self.cloud_id / 100)
        self.points = read_xyz(filename)
        self.feature: list[float] = []

    def compute_features(self) -> None:
        pts = self.points
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        n_points = len(pts)
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        zmin, zmax = np.min(z), np.max(z)

        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin

        height = float(dz)

        root = pts[[np.argmin(z)]]
        top = pts[[np.argmax(z)]]

        kd_tree_2d = KDTree(pts[:, :2], leaf_size=5)
        kd_tree_3d = KDTree(pts, leaf_size=5)

        count = kd_tree_2d.query_radius(root[:, :2], r=0.2, count_only=True)
        root_density = float(count[0] / max(n_points, 1))

        try:
            hull_2d = ConvexHull(pts[:, :2])
            hull_area = float(hull_2d.volume)
            hull_perimeter = float(hull_2d.area)
        except QhullError:
            hull_area = 0.0
            hull_perimeter = 0.0

        area = hull_area
        shape_index = float(hull_area / (hull_perimeter + 1e-5))
        circularity = float(4.0 * np.pi * hull_area / (hull_perimeter**2 + 1e-5))

        k_top = min(max(int(n_points * 0.01), 30), n_points)
        idx = kd_tree_3d.query(top, k=k_top, return_distance=False)
        idx = np.squeeze(idx, axis=0)
        neighbours = pts[idx, :]

        cov = np.cov(neighbours.T)
        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.sort(np.real(eigvals))

        linearity = float((eigvals[2] - eigvals[1]) / (eigvals[2] + 1e-5))
        sphericity = float(eigvals[0] / (eigvals[2] + 1e-5))

        slenderness = float(dz / (max(dx, dy) + 1e-5))
        length_height_ratio = float(max(dx, dy) / (dz + 1e-5))
        footprint_density = float(n_points / (hull_area + 1e-5))

        self.feature = [
            height,
            root_density,
            area,
            shape_index,
            linearity,
            sphericity,
            slenderness,
            length_height_ratio,
            circularity,
            footprint_density,
        ]


def read_xyz(filename: str | Path) -> np.ndarray:
    points = []
    with open(filename, "r", encoding="utf-8") as f_input:
        for line in f_input:
            xyz = [float(v) for v in line.split()]
            points.append(xyz)
    return np.asarray(points, dtype=np.float32)


def feature_preparation(
    data_path: str | Path,
    data_file: str | Path = "data.txt",
    force_recompute: bool = False,
) -> None:
    data_path = Path(data_path)
    data_file = Path(data_file)

    if data_file.exists() and not force_recompute:
        return

    files = sorted(p for p in data_path.iterdir() if p.is_file() and p.suffix.lower() == ".xyz")
    input_data: list[list[float]] = []

    for file_path in tqdm(files, total=len(files)):
        obj = UrbanObject(file_path)
        obj.compute_features()
        row = [obj.cloud_id, obj.label] + obj.feature
        input_data.append(row)

    outputs = np.asarray(input_data, dtype=np.float32)
    header = "ID,label," + ",".join(FEATURE_NAMES)
    np.savetxt(
        data_file,
        outputs,
        fmt="%10.6f",
        delimiter=",",
        newline="\n",
        header=header,
    )
