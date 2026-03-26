from pathlib import Path

from features import feature_preparation
from classifiers import data_loading, tune_svm, tune_rf
from visualize_features import scatter_two_features


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "pointclouds-500"
    data_file = base_dir / "data.txt"

    print("Start preparing features")
    feature_preparation(
        data_path=str(data_path),
        data_file=str(data_file),
        force_recompute=True
    )

    print("Start loading data")
    ID, X, y = data_loading(str(data_file))

    print("Visualize features")
    # Current feature indices:
    # 0 height
    # 1 root_density
    # 2 area
    # 3 shape_index
    # 4 linearity
    # 5 sphericity
    # 6 slenderness
    # 7 length_height_ratio
    # 8 circularity
    # 9 footprint_density

    scatter_two_features(X, y, feat_x=6, feat_y=7)               # slenderness vs length_height_ratio
    scatter_two_features(X, y, feat_x=8, feat_y=9)               # circularity vs footprint_density
    scatter_two_features(X, y, feat_x=1, feat_y=2, log_y=True)   # root_density vs area

    print("Start SVM tuning and classification")
    svm_results = tune_svm(X, y, test_size=0.4, random_state=42)

    print("Start RF tuning and classification")
    rf_results = tune_rf(X, y, test_size=0.4, random_state=42)

    print("\n Our recommended models:")
    print("Best SVM params:", svm_results["best_params"])
    print("Best RF params:", rf_results["best_params"])