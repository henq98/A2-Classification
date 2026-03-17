from features import feature_preparation
from classifiers import data_loading, SVM_classification, RF_classification
from visualize_features import scatter_two_features


if __name__ == '__main__':
    path = "/Users/macbook-mathijs/CME Master/Machine learning/Assignment 2/A2-Classification/pointclouds-500"

    print('Start preparing features')
    feature_preparation(data_path=path, data_file='data.txt', force_recompute=True)

    print('Start loading data from the local file')
    ID, X, y = data_loading('data.txt')

    print('Visualize features')
    scatter_two_features(X, y, feat_x=8, feat_y=18)   # slenderness vs top_roughness
    scatter_two_features(X, y, feat_x=7, feat_y=6)    # elongation_xy vs circularity
    scatter_two_features(X, y, feat_x=13, feat_y=20)  # length_height_ratio vs top_to_bottom_area_ratio

    print('Start full SVM classification')
    svm_model, svm_acc = SVM_classification(X, y)

    print('Start full RF classification')
    rf_model, rf_acc, rf_importances, rf_ranking = RF_classification(X, y)

    top8 = rf_ranking[:8].tolist()
    top4 = rf_ranking[:4].tolist()

    print("\nStart SVM with top 8 RF-ranked features")
    SVM_classification(X, y, feature_indices=top8)

    print("\nStart RF with top 8 RF-ranked features")
    RF_classification(X, y, feature_indices=top8)

    print("\nStart SVM with top 4 RF-ranked features")
    SVM_classification(X, y, feature_indices=top4)

    print("\nStart RF with top 4 RF-ranked features")
    RF_classification(X, y, feature_indices=top4)