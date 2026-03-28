import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def _subset_features(X, feature_indices=None):
    if feature_indices is None:
        return X
    return X[:, feature_indices]


def compute_learning_curve(
    estimator,
    X,
    y,
    feature_indices=None,
    train_ratios=None,
    n_repeats=10,
    random_state=42,
):
    """
    Manual learning-curve implementation.

    For each train:test ratio (1:9, 2:8, ..., 9:1):
    1) split into training and testing sets
    2) fit classifier on training set
    3) compute training error
    4) compute testing error
    5) repeat several times and average

    This explicitly implements the required steps and does NOT use
    sklearn's learning_curve convenience function.
    """
    X = _subset_features(X, feature_indices)
    y = np.asarray(y)

    if train_ratios is None:
        train_ratios = [i / 10.0 for i in range(1, 10)]  # 0.1 ... 0.9

    results = []

    for ratio in train_ratios:
        train_errors = []
        test_errors = []
        n_trains = []
        n_tests = []

        for repeat in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=ratio,
                stratify=y,
                random_state=random_state + repeat,
            )

            model = clone(estimator)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_err = 1.0 - accuracy_score(y_train, y_train_pred)
            test_err = 1.0 - accuracy_score(y_test, y_test_pred)

            train_errors.append(train_err)
            test_errors.append(test_err)
            n_trains.append(len(y_train))
            n_tests.append(len(y_test))

        results.append(
            {
                "train_ratio": ratio,
                "n_train": int(round(np.mean(n_trains))),
                "n_test": int(round(np.mean(n_tests))),
                "train_error_mean": float(np.mean(train_errors)),
                "train_error_std": float(np.std(train_errors, ddof=1)) if len(train_errors) > 1 else 0.0,
                "test_error_mean": float(np.mean(test_errors)),
                "test_error_std": float(np.std(test_errors, ddof=1)) if len(test_errors) > 1 else 0.0,
            }
        )

    return results


def print_learning_curve_results(results, model_name):
    print(f"\n===== LEARNING CURVE RESULTS: {model_name} =====")
    print(
        f"{'Train ratio':>11s} {'n_train':>8s} {'n_test':>7s} "
        f"{'Train error':>15s} {'Test error':>15s}"
    )
    for row in results:
        print(
            f"{row['train_ratio']:11.1f} "
            f"{row['n_train']:8d} "
            f"{row['n_test']:7d} "
            f"{row['train_error_mean']:13.4f}±{row['train_error_std']:.4f} "
            f"{row['test_error_mean']:13.4f}±{row['test_error_std']:.4f}"
        )


def plot_learning_curve_error(results, model_name, save_path=None, show=True):
    x = [row["n_train"] for row in results]
    train_y = [row["train_error_mean"] for row in results]
    test_y = [row["test_error_mean"] for row in results]

    plt.figure(figsize=(9, 6))
    plt.plot(x, train_y, marker="o", label="Training")
    plt.plot(x, test_y, marker="s", label="Testing")
    plt.xlabel("Number of training samples")
    plt.ylabel("Classification error rate")
    plt.title(f"Learning curve ({model_name}, error rate)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()