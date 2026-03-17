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

        height = dz
        mean_z = np.mean(z)
        std_z = np.std(z)

        root = pts[[np.argmin(z)]]
        top = pts[[np.argmax(z)]]

        kd_tree_2d = KDTree(pts[:, :2], leaf_size=5)
        kd_tree_3d = KDTree(pts, leaf_size=5)

        # Root density
        radius_root = 0.2
        count = kd_tree_2d.query_radius(root[:, :2], r=radius_root, count_only=True)
        root_density = count[0] / max(n_points, 1)

        # 2D convex hull features
        try:
            hull_2d = ConvexHull(pts[:, :2])
            hull_area = float(hull_2d.volume)
            hull_perimeter = float(hull_2d.area)
        except QhullError:
            hull_area = 0.0
            hull_perimeter = 0.0

        shape_index = hull_area / (hull_perimeter + 1e-5)
        circularity = 4.0 * np.pi * hull_area / (hull_perimeter**2 + 1e-5)

        elongation_xy = max(dx, dy) / (min(dx, dy) + 1e-5)
        slenderness = dz / (max(dx, dy) + 1e-5)
        rectangularity = hull_area / (dx * dy + 1e-5)
        footprint_density = n_points / (hull_area + 1e-5)
        bbox_volume = dx * dy * dz
        point_density_3d = n_points / (bbox_volume + 1e-5)

        # New targeted feature: long-and-low objects (fences)
        length_height_ratio = max(dx, dy) / (dz + 1e-5)

        # Height distribution
        z_norm = (z - zmin) / (dz + 1e-5)
        lower_fraction = np.mean(z_norm <= 0.33)
        middle_fraction = np.mean((z_norm > 0.33) & (z_norm <= 0.66))
        upper_fraction = np.mean(z_norm > 0.66)

        # Vertical entropy
        hist, _ = np.histogram(z_norm, bins=5, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float64) / max(np.sum(hist), 1)
        z_entropy = -np.sum(hist * np.log(hist + 1e-10))

        # Top region roughness
        top_mask = z >= np.percentile(z, 90)
        top_points = pts[top_mask]
        if len(top_points) >= 3:
            A = np.column_stack((top_points[:, 0], top_points[:, 1], np.ones(len(top_points))))
            coeffs, _, _, _ = np.linalg.lstsq(A, top_points[:, 2], rcond=None)
            z_fit = A @ coeffs
            top_roughness = np.sqrt(np.mean((top_points[:, 2] - z_fit) ** 2))
        else:
            top_roughness = 0.0

        # Top density
        top_density = len(top_points) / (hull_area + 1e-5)

        # New targeted feature: canopy-vs-base behavior
        bottom_mask = z <= np.percentile(z, 20)

        def safe_area(p_subset):
            if len(p_subset) < 3:
                return 0.0
            try:
                return float(ConvexHull(p_subset[:, :2]).volume)
            except QhullError:
                return 0.0

        top_area = safe_area(top_points)
        bottom_area = safe_area(pts[bottom_mask])
        top_to_bottom_area_ratio = top_area / (bottom_area + 1e-5)

        # Local top-neighborhood covariance features
        k_top = min(max(int(n_points * 0.01), 100), n_points)
        idx = kd_tree_3d.query(top, k=k_top, return_distance=False)
        idx = np.squeeze(idx, axis=0)
        neighbours = pts[idx, :]

        cov = np.cov(neighbours.T)
        w, _ = np.linalg.eig(cov)
        w = np.sort(np.real(w))

        linearity = (w[2] - w[1]) / (w[2] + 1e-5)
        planarity = (w[1] - w[0]) / (w[2] + 1e-5)
        sphericity = w[0] / (w[2] + 1e-5)
        anisotropy = (w[2] - w[0]) / (w[2] + 1e-5)
        curvature = w[0] / (w[0] + w[1] + w[2] + 1e-5)

        self.feature = [
            height,
            mean_z,
            std_z,
            root_density,
            hull_area,
            shape_index,
            circularity,
            elongation_xy,
            slenderness,
            rectangularity,
            footprint_density,
            bbox_volume,
            point_density_3d,
            length_height_ratio,
            lower_fraction,
            middle_fraction,
            upper_fraction,
            z_entropy,
            top_roughness,
            top_density,
            top_to_bottom_area_ratio,
            linearity,
            planarity,
            sphericity,
            anisotropy,
            curvature,
        ]


def read_xyz(filenm):
    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = [float(i) for i in line.split()]
            points.append(p)
    return np.array(points).astype(np.float32)


def feature_preparation(data_path, data_file='data.txt', force_recompute=False):
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
        "height,mean_z,std_z,root_density,area,shape_index,circularity,"
        "elongation_xy,slenderness,rectangularity,footprint_density,"
        "bbox_volume,point_density_3d,length_height_ratio,"
        "lower_fraction,middle_fraction,upper_fraction,z_entropy,"
        "top_roughness,top_density,top_to_bottom_area_ratio,"
        "linearity,planarity,sphericity,anisotropy,curvature"
    )

    np.savetxt(
        data_file,
        outputs,
        fmt='%10.6f',
        delimiter=',',
        newline='\n',
        header=data_header
    )