import math
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull, QhullError
from tqdm import tqdm
from os.path import exists, join, basename
from os import listdir


class urban_object:
    def __init__(self, filenm):
        self.cloud_name = basename(filenm)[:-4]
        self.cloud_ID = int(self.cloud_name)
        self.label = math.floor(self.cloud_ID / 100)
        self.points = read_xyz(filenm)
        self.feature = []

    def compute_features(self):
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

        # -----------------------------
        # Reference/professor-style features
        # -----------------------------
        height = dz

        root = pts[[np.argmin(z)]]
        top = pts[[np.argmax(z)]]

        kd_tree_2d = KDTree(pts[:, :2], leaf_size=5)
        kd_tree_3d = KDTree(pts, leaf_size=5)

        radius_root = 0.2
        count = kd_tree_2d.query_radius(root[:, :2], r=radius_root, count_only=True)
        root_density = count[0] / max(n_points, 1)

        try:
            hull_2d = ConvexHull(pts[:, :2])
            hull_area = float(hull_2d.volume)
            hull_perimeter = float(hull_2d.area)
        except QhullError:
            hull_area = 0.0
            hull_perimeter = 0.0

        area = hull_area
        shape_index = hull_area / (hull_perimeter + 1e-5)

        k_top = min(max(int(n_points * 0.01), 100), n_points)
        idx = kd_tree_3d.query(top, k=k_top, return_distance=False)
        idx = np.squeeze(idx, axis=0)
        neighbours = pts[idx, :]

        cov = np.cov(neighbours.T)
        w, _ = np.linalg.eig(cov)
        w = np.sort(np.real(w))

        linearity = (w[2] - w[1]) / (w[2] + 1e-5)
        sphericity = w[0] / (w[2] + 1e-5)

        # -----------------------------
        # strongest features chosen from feature selection in previous runs
        # -----------------------------
        circularity = 4.0 * np.pi * hull_area / (hull_perimeter**2 + 1e-5)
        slenderness = dz / (max(dx, dy) + 1e-5)
        length_height_ratio = max(dx, dy) / (dz + 1e-5)
        footprint_density = n_points / (hull_area + 1e-5)

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


def read_xyz(filenm):
    points = []
    with open(filenm, "r") as f_input:
        for line in f_input:
            p = [float(i) for i in line.split()]
            points.append(p)
    return np.array(points).astype(np.float32)


def feature_preparation(data_path, data_file="data.txt", force_recompute=False):
    if exists(data_file) and not force_recompute:
        return

    files = sorted(listdir(data_path))
    input_data = []

    for file_i in tqdm(files, total=len(files)):
        file_name = join(data_path, file_i)
        i_object = urban_object(filenm=file_name)
        i_object.compute_features()
        i_data = [i_object.cloud_ID, i_object.label] + i_object.feature
        input_data.append(i_data)

    outputs = np.array(input_data).astype(np.float32)

    data_header = (
        "ID,label,"
        "height,root_density,area,shape_index,linearity,sphericity,"
        "slenderness,length_height_ratio,circularity,footprint_density"
    )

    np.savetxt(
        data_file,
        outputs,
        fmt="%10.6f",
        delimiter=",",
        newline="\n",
        header=data_header,
    )