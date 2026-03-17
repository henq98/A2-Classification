from features import feature_preparation
from classifiers import data_loading, SVM_classification, RF_classification
from visualize_features import scatter_two_features


if __name__ == '__main__':
    path = "/Users/macbook-mathijs/CME Master/Machine learning/Assignment 2/A2-Classification/pointclouds-500"

    print('Start preparing features')
    feature_preparation(data_path=path)

    print('Start loading data from the local file')
    ID, X, y = data_loading()

    print('Visualize features')
    scatter_two_features(X, y, feat_x=1, feat_y=2)

    print('Start SVM classification')
    SVM_classification(X, y)

    print('Start RF classification')
    RF_classification(X, y)
    
