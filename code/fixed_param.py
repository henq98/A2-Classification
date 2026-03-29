import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    'height',
    'root_density',
    'area',
    'shape_index',
    'linearity',
    'sphericity',
    'slenderness',
    'length_height_ratio',
    'circularity',
    'footprint_density',
]

CLASS_NAMES = ["building", "car", "fence", "pole", "tree"]



def data_loading(data_file='data.txt'):
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=',', comments='#')
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)
    return ID, X, y



def select_features(X, feature_indices=None):
    if feature_indices is None:
        return X
    return X[:, feature_indices]



def get_feature_names(feature_indices=None):
    if feature_indices is None:
        return FEATURE_NAMES
    return [FEATURE_NAMES[i] for i in feature_indices]



def SVM_classification(X, y, feature_indices=None, kernel="rbf", C=50):
    X = select_features(X, feature_indices)
    selected_feature_names = get_feature_names(feature_indices)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm.SVC(kernel=kernel, C=C))
    ])

    pipeline.fit(X_train, y_train)
    y_preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_preds)
    print("\n===== SVM RESULTS =====")
    print("Used features:", selected_feature_names)
    print("Fixed parameters:", {"svm__kernel": kernel, "svm__C": C})
    print("Test accuracy: %5.3f" % acc)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_preds))
    print("Classification report:")
    print(classification_report(y_test, y_preds, target_names=CLASS_NAMES, digits=3))

    return pipeline, acc, {"svm__kernel": kernel, "svm__C": C}


def RF_classification(X, y, feature_indices=None, n_estimators=100, max_depth=None):
    X = select_features(X, feature_indices)
    selected_feature_names = get_feature_names(feature_indices)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    rf.fit(X_train, y_train)
    y_preds = rf.predict(X_test)

    acc = accuracy_score(y_test, y_preds)
    print("\n===== RF RESULTS =====")
    print("Used features:", selected_feature_names)
    print("Fixed parameters:", {
        "n_estimators": n_estimators,
        "max_depth": max_depth
    })
    print("Test accuracy: %5.3f" % acc)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_preds))
    print("Classification report:")
    print(classification_report(y_test, y_preds, target_names=CLASS_NAMES, digits=3))

    importances = rf.feature_importances_
    ranking = np.argsort(importances)[::-1]

    print("\nRF feature importances for the used feature set:")
    for i in ranking:
        print(f"{selected_feature_names[i]:20s} {importances[i]:.4f}")

    return rf, acc, {"n_estimators": n_estimators, "max_depth": max_depth}, importances, ranking