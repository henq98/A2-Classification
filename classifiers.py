import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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



def SVM_classification(X, y, feature_indices=None):
    X = select_features(X, feature_indices)
    selected_feature_names = get_feature_names(feature_indices)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm.SVC())
    ])

    param_grid = [
        {
            "svm__kernel": ["rbf"],
            "svm__C": [5, 10, 15, 30, 50, 100],
            "svm__gamma": ["scale", 0.01, 0.03, 0.05, 0.1]
        },
        {
            "svm__kernel": ["linear"],
            "svm__C": [5, 10, 15, 30, 50, 100]
        },
        {
            "svm__kernel": ["poly"],
            "svm__C": [5, 10, 15, 30, 50, 100],
            "svm__degree": [1, 2, 3],
            "svm__gamma": ["scale", 0.01, 0.03, 0.05, 0.10]
        }
    ]

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
    print("Used features:", selected_feature_names)
    print("Best parameters:", grid.best_params_)
    print("Cross-val best score: %5.3f" % grid.best_score_)
    print("Test accuracy: %5.3f" % acc)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_preds))
    print("Classification report:")
    print(classification_report(y_test, y_preds, target_names=CLASS_NAMES, digits=3))

    return best_model, acc, grid.best_params_, grid.best_score_



def RF_classification(X, y, feature_indices=None):
    X = select_features(X, feature_indices)
    selected_feature_names = get_feature_names(feature_indices)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [150, 200, 300, 400],
        "max_depth": [None, 15, 20, 30],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
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
    print("Used features:", selected_feature_names)
    print("Best parameters:", grid.best_params_)
    print("Cross-val best score: %5.3f" % grid.best_score_)
    print("Test accuracy: %5.3f" % acc)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_preds))
    print("Classification report:")
    print(classification_report(y_test, y_preds, target_names=CLASS_NAMES, digits=3))

    importances = best_model.feature_importances_
    ranking = np.argsort(importances)[::-1]

    print("\nRF feature importances for the used feature set:")
    for i in ranking:
        print(f"{selected_feature_names[i]:20s} {importances[i]:.4f}")

    return best_model, acc, grid.best_params_, grid.best_score_, importances, ranking

