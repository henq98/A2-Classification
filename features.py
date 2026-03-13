import math
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull
from tqdm import tqdm
from os.path import exists, join, basename
from os import listdir


class urban_object:
    def __init__(self, filenm):
        self.cloud_name = basename(filenm)[:-4]
        self.cloud_ID = int(self.cloud_name)
        self.label = math.floor(1.0 * self.cloud_ID / 100)
        self.points = read_xyz(filenm)
        self.feature = []

    def compute_features(self):
        zmin = np.min(self.points[:, 2])
        zmax = np.max(self.points[:, 2])
        height = zmax - zmin
        self.feature.append(height)

        root = self.points[[np.argmin(self.points[:, 2])]]
        top = self.points[[np.argmax(self.points[:, 2])]]

        kd_tree_2d = KDTree(self.points[:, :2], leaf_size=5)
        kd_tree_3d = KDTree(self.points, leaf_size=5)

        radius_root = 0.2
        count = kd_tree_2d.query_radius(root[:, :2], r=radius_root, count_only=True)
        root_density = 1.0 * count[0] / len(self.points)
        self.feature.append(root_density)

        hull_2d = ConvexHull(self.points[:, :2])
        hull_area = hull_2d.volume
        self.feature.append(hull_area)

        hull_perimeter = hull_2d.area
        shape_index = 1.0 * hull_area / (hull_perimeter + 1e-5)
        self.feature.append(shape_index)

        xmin, ymin = np.min(self.points[:, 0]), np.min(self.points[:, 1])
        xmax, ymax = np.max(self.points[:, 0]), np.max(self.points[:, 1])
        dx = xmax - xmin
        dy = ymax - ymin

        elongation_xy = max(dx, dy) / (min(dx, dy) + 1e-5)
        slenderness = height / (max(dx, dy) + 1e-5)
        circularity = 4 * np.pi * hull_area / (hull_perimeter**2 + 1e-5)
        std_z = np.std(self.points[:, 2])

        self.feature += [elongation_xy, slenderness, circularity, std_z]

        k_top = min(max(int(len(self.points) * 0.005), 100), len(self.points))
        idx = kd_tree_3d.query(top, k=k_top, return_distance=False)
        idx = np.squeeze(idx, axis=0)
        neighbours = self.points[idx, :]

        cov = np.cov(neighbours.T)
        w, _ = np.linalg.eig(cov)
        w = np.sort(np.real(w))

        linearity = (w[2] - w[1]) / (w[2] + 1e-5)
        sphericity = w[0] / (w[2] + 1e-5)
        self.feature += [linearity, sphericity]


def read_xyz(filenm):
    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = [float(i) for i in line.split()]
            points.append(p)
    return np.array(points).astype(np.float32)


def feature_preparation(data_path, data_file='data.txt'):
    if exists(data_file):
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
        'ID,label,height,root_density,area,shape_index,'
        'elongation_xy,slenderness,circularity,std_z,linearity,sphericity'
    )
    np.savetxt(data_file, outputs, fmt='%10.5f', delimiter=',', newline='\n', header=data_header)