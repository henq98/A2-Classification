import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "height",
    "root_density",
    "area",
    "shape_index",
    "linearity",
    "sphericity",
    "slenderness",
    "length_height_ratio",
    "circularity",
    "footprint_density",
]


def data_loading(data_file="data.txt"):
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=",", comments="#")
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)
    return ID, X, y


def subset_features(X, feature_indices=None):
    if feature_indices is None:
        return X
    return X[:, feature_indices]


def print_results(title, y_test, y_pred, best_params=None, cv_score=None):
    acc = accuracy_score(y_test, y_pred)
    err = 1.0 - acc

    print(f"\n===== {title} =====")
    if best_params is not None:
        print("Best parameters:", best_params)
    if cv_score is not None:
        print("Best cross-val accuracy: %.3f" % cv_score)
        print("Best cross-val error rate: %.3f" % (1.0 - cv_score))

    print("Test accuracy: %.3f" % acc)
    print("Test error rate: %.3f" % err)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))
    return acc


def get_split(X, y, test_size=0.4, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def tune_svm(X, y, feature_indices=None, test_size=0.4, random_state=42):
    X = subset_features(X, feature_indices)
    X_train, X_test, y_train, y_test = get_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm.SVC())
    ])

    param_grid = [
        {
            "svm__kernel": ["linear"],
            "svm__C": [0.01, 0.1, 1, 10, 30, 50, 100]
        },
        {
            "svm__kernel": ["rbf"],
            "svm__C": [0.1, 1, 5, 10, 15, 30, 50, 100],
            "svm__gamma": ["scale", 0.01, 0.03, 0.05, 0.1]
        },
        {
            "svm__kernel": ["poly"],
            "svm__C": [0.1, 1, 10, 50],
            "svm__degree": [2, 3],
            "svm__gamma": ["scale", 0.01, 0.1]
        }
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = print_results(
        title="SVM RESULTS",
        y_test=y_test,
        y_pred=y_pred,
        best_params=grid.best_params_,
        cv_score=grid.best_score_
    )

    return {
        "model": best_model,
        "accuracy": acc,
        "error_rate": 1.0 - acc,
        "best_params": grid.best_params_,
        "cv_score": grid.best_score_,
        "cv_results": grid.cv_results_,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def tune_rf(X, y, feature_indices=None, test_size=0.4, random_state=42):
    X = subset_features(X, feature_indices)
    X_train, X_test, y_train, y_test = get_split(
        X, y, test_size=test_size, random_state=random_state
    )

    rf = RandomForestClassifier(random_state=random_state)

    param_grid = {
        "n_estimators": [100, 150, 200, 300, 400],
        "max_depth": [None, 10, 20],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = print_results(
        title="RF RESULTS",
        y_test=y_test,
        y_pred=y_pred,
        best_params=grid.best_params_,
        cv_score=grid.best_score_
    )

    importances = best_model.feature_importances_
    ranking = np.argsort(importances)[::-1]

    print("\nTop RF feature importances:")
    for i in ranking:
        print(f"{FEATURE_NAMES[i]:25s} {importances[i]:.4f}")

    return {
        "model": best_model,
        "accuracy": acc,
        "error_rate": 1.0 - acc,
        "best_params": grid.best_params_,
        "cv_score": grid.best_score_,
        "cv_results": grid.cv_results_,
        "importances": importances,
        "ranking": ranking,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }