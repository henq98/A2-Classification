import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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

    X = _subset_features(X, feature_indices)
    y = np.asarray(y)

    if train_ratios is None:
        train_ratios = [i / 10.0 for i in range(1, 10)]  # 0.1, 0.2, adn so on

    results = []

    for ratio in train_ratios:
        train_accs = []
        test_accs = []
        train_sizes = []
        test_sizes = []

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

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            train_accs.append(train_acc)
            test_accs.append(test_acc)
            train_sizes.append(len(y_train))
            test_sizes.append(len(y_test))

        results.append(
            {
                "train_ratio": ratio,
                "n_train": int(round(np.mean(train_sizes))),
                "n_test": int(round(np.mean(test_sizes))),
                "train_oa_mean": float(np.mean(train_accs)),
                "train_oa_std": float(np.std(train_accs, ddof=1)) if len(train_accs) > 1 else 0.0,
                "test_oa_mean": float(np.mean(test_accs)),
                "test_oa_std": float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0,
                "train_error_mean": float(1.0 - np.mean(train_accs)),
                "test_error_mean": float(1.0 - np.mean(test_accs)),
            }
        )

    return results


def print_learning_curve_results(results, model_name):
    print(f"\n===== LEARNING CURVE RESULTS: {model_name} =====")
    print(
        f"{'Train ratio':>11s} {'n_train':>8s} {'n_test':>7s} "
        f"{'Train OA':>12s} {'Test OA':>12s} {'Train err':>12s} {'Test err':>11s}"
    )
    for row in results:
        print(
            f"{row['train_ratio']:11.1f} {row['n_train']:8d} {row['n_test']:7d} "
            f"{row['train_oa_mean']:10.3f}±{row['train_oa_std']:.3f} "
            f"{row['test_oa_mean']:10.3f}±{row['test_oa_std']:.3f} "
            f"{row['train_error_mean']:12.3f} {row['test_error_mean']:11.3f}"
        )


def plot_learning_curve(
    results,
    model_name,
    metric="oa",
    save_path=None,
    show=True,
):

    x = [row["n_train"] for row in results]

    if metric == "error":
        train_y = [row["train_error_mean"] for row in results]
        test_y = [row["test_error_mean"] for row in results]
        ylabel = "Classification error rate"
        title = f"Learning curve ({model_name}, error rate)"
    else:
        train_y = [row["train_oa_mean"] for row in results]
        test_y = [row["test_oa_mean"] for row in results]
        ylabel = "Overall accuracy"
        title = f"Learning curve ({model_name}, OA)"

    plt.figure(figsize=(7, 5))
    plt.plot(x, train_y, marker="o", label="Training")
    plt.plot(x, test_y, marker="s", label="Testing")
    plt.xlabel("Number of training samples")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
