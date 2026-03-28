from pathlib import Path
import numpy as np

from features import feature_preparation
from classifiers import data_loading, tune_svm, tune_rf
from visualize_features import scatter_two_features
from learning_curves import (
    compute_learning_curve,
    print_learning_curve_results,
    plot_learning_curve_error,
)


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

CLASS_NAMES = ["building", "car", "fence", "pole", "tree"]

TRAIN_RATIOS = [i / 10.0 for i in range(1, 10)]
N_REPEATS = 10


def scatter_ratio_score(X_subset, y):
    """
    Scatter-matrix score = trace(S_B) / trace(S_W)
    """
    classes = np.unique(y)
    overall_mean = np.mean(X_subset, axis=0, keepdims=True)

    n_features = X_subset.shape[1]
    Sw = np.zeros((n_features, n_features), dtype=np.float64)
    Sb = np.zeros((n_features, n_features), dtype=np.float64)

    for c in classes:
        Xc = X_subset[y == c]
        if len(Xc) == 0:
            continue

        mean_c = np.mean(Xc, axis=0, keepdims=True)
        centered = Xc - mean_c
        Sw += centered.T @ centered

        mean_diff = mean_c - overall_mean
        Sb += len(Xc) * (mean_diff.T @ mean_diff)

    return float(np.trace(Sb) / (np.trace(Sw) + 1e-12))


def forward_select_features(X, y, k=4):
    """
    Greedy forward search to select 4 features.
    """
    selected = []
    remaining = list(range(X.shape[1]))

    print("\n===== FEATURE SELECTION (forward search) =====")

    for step in range(k):
        best_feature = None
        best_score = -np.inf

        for feat_idx in remaining:
            candidate = selected + [feat_idx]
            score = scatter_ratio_score(X[:, candidate], y)

            if score > best_score:
                best_score = score
                best_feature = feat_idx

        selected.append(best_feature)
        remaining.remove(best_feature)

        print(
            f"Step {step + 1}: add feature {best_feature} "
            f"({FEATURE_NAMES[best_feature]}) -> score = {best_score:.6f}"
        )

    print("\nSelected feature indices:", selected)
    print("Selected feature names:")
    for idx in selected:
        print(f"  - {idx}: {FEATURE_NAMES[idx]}")

    return selected


def print_final_error_comparison(svm_results, rf_results):
    print("\n===== FINAL MODEL COMPARISON (ERROR RATE) =====")
    print(f"SVM error rate: {svm_results['error_rate']:.4f}")
    print(f"RF  error rate: {rf_results['error_rate']:.4f}")

    if svm_results["error_rate"] < rf_results["error_rate"]:
        print("Best classifier based on error rate: SVM")
    elif rf_results["error_rate"] < svm_results["error_rate"]:
        print("Best classifier based on error rate: RF")
    else:
        print("Both classifiers have the same error rate.")


def print_error_analysis(name, y_test, y_pred):
    print(f"\n===== ERROR ANALYSIS: {name} =====")
    conf = np.zeros((5, 5), dtype=int)

    for yt, yp in zip(y_test, y_pred):
        conf[int(yt), int(yp)] += 1

    print("Confusion matrix:")
    print(conf)

    for i in range(5):
        total = np.sum(conf[i, :])
        correct = conf[i, i]
        mistakes = total - correct
        err_rate = mistakes / total if total > 0 else 0.0
        print(
            f"Class {i} ({CLASS_NAMES[i]}): "
            f"correct = {correct}, mistakes = {mistakes}, class error = {err_rate:.3f}"
        )

    print("\nMost common confusion pairs:")
    conf_no_diag = conf.copy()
    np.fill_diagonal(conf_no_diag, 0)

    pairs = []
    for i in range(5):
        for j in range(5):
            if i != j and conf_no_diag[i, j] > 0:
                pairs.append((conf_no_diag[i, j], i, j))

    pairs.sort(reverse=True)

    if len(pairs) == 0:
        print("No misclassifications.")
    else:
        for count, i, j in pairs[:5]:
            print(f"  {CLASS_NAMES[i]} -> {CLASS_NAMES[j]} : {count}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "pointclouds-500"
    data_file = base_dir / "data.txt"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print("Start preparing features")
    feature_preparation(
        data_path=str(data_path),
        data_file=str(data_file),
        force_recompute=True,
    )

    print("Start loading data")
    ID, X, y = data_loading(str(data_file))

    print("Visualize features")
    scatter_two_features(X, y, feat_x=6, feat_y=7)               # slenderness vs length_height_ratio
    scatter_two_features(X, y, feat_x=8, feat_y=9)               # circularity vs footprint_density
    scatter_two_features(X, y, feat_x=1, feat_y=2, log_y=True)   # root_density vs area

    selected_indices = forward_select_features(X, y, k=4)

    print("\nStart SVM tuning and classification")
    svm_results = tune_svm(
        X,
        y,
        feature_indices=selected_indices,
        test_size=0.4,
        random_state=42,
    )

    print("\nStart RF tuning and classification")
    rf_results = tune_rf(
        X,
        y,
        feature_indices=selected_indices,
        test_size=0.4,
        random_state=42,
    )

    print_final_error_comparison(svm_results, rf_results)

    print_error_analysis("SVM", svm_results["y_test"], svm_results["y_pred"])
    print_error_analysis("RF", rf_results["y_test"], rf_results["y_pred"])

    svm_model_selected = svm_results["model"]
    rf_model_selected = rf_results["model"]

    print("\nGenerate learning curve for final SVM model")
    svm_curve = compute_learning_curve(
        estimator=svm_model_selected,
        X=X,
        y=y,
        feature_indices=selected_indices,
        train_ratios=TRAIN_RATIOS,
        n_repeats=N_REPEATS,
        random_state=42,
    )
    print_learning_curve_results(svm_curve, "SVM")
    plot_learning_curve_error(
        svm_curve,
        model_name="SVM",
        save_path=results_dir / "learning_curve_svm_error.png",
        show=False,
    )

    print("\nGenerate learning curve for final RF model")
    rf_curve = compute_learning_curve(
        estimator=rf_model_selected,
        X=X,
        y=y,
        feature_indices=selected_indices,
        train_ratios=TRAIN_RATIOS,
        n_repeats=N_REPEATS,
        random_state=42,
    )
    print_learning_curve_results(rf_curve, "RF")
    plot_learning_curve_error(
        rf_curve,
        model_name="RF",
        save_path=results_dir / "learning_curve_rf_error.png",
        show=False,
    )

    print("\nDone.")
    print("Saved results in:", results_dir)