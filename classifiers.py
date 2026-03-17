import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "height", "mean_z", "std_z", "root_density", "area", "shape_index", "circularity",
    "elongation_xy", "slenderness", "rectangularity", "footprint_density",
    "bbox_volume", "point_density_3d", "lower_fraction", "middle_fraction",
    "upper_fraction", "top_roughness", "linearity", "planarity",
    "sphericity", "anisotropy", "curvature"
]


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


def SVM_classification(X, y, feature_indices=None):
    X = select_features(X, feature_indices)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm.SVC())
    ])

    param_grid = {
        "svm__kernel": ["rbf", "linear", "poly"],
        "svm__C": [0.1, 1, 10, 50, 100],
        "svm__gamma": ["scale", 0.01, 0.1, 1],
        "svm__degree": [2, 3]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_preds = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_preds)
    print("\n===== SVM RESULTS =====")
    print("Best parameters:", grid.best_params_)
    print("Cross-val best score: %5.3f" % grid.best_score_)
    print("Test accuracy: %5.3f" % acc)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_preds))
    print("Classification report:")
    print(classification_report(y_test, y_preds, digits=3))

    return best_model, acc


def RF_classification(X, y, feature_indices=None):
    X = select_features(X, feature_indices)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }

    rf = RandomForestClassifier(random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        rf,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_preds = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_preds)
    print("\n===== RF RESULTS =====")
    print("Best parameters:", grid.best_params_)
    print("Cross-val best score: %5.3f" % grid.best_score_)
    print("Test accuracy: %5.3f" % acc)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_preds))
    print("Classification report:")
    print(classification_report(y_test, y_preds, digits=3))

    importances = best_model.feature_importances_
    ranking = np.argsort(importances)[::-1]

    print("\nTop RF feature importances:")
    for i in ranking[:10]:
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
        print(f"{name:20s} {importances[i]:.4f}")

    return best_model, acc, importances