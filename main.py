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
    scatter_two_features(X, y, feat_x=8, feat_y=16)   # slenderness vs top_roughness
    scatter_two_features(X, y, feat_x=7, feat_y=6)    # elongation_xy vs circularity
    scatter_two_features(X, y, feat_x=3, feat_y=4, log_y=True)  # root_density vs area

    print('Start SVM classification')
    svm_model, svm_acc = SVM_classification(X, y)

    print('Start RF classification')
    rf_model, rf_acc, rf_importances = RF_classification(X, y)