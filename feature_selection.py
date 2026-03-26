import numpy as np
from sklearn.preprocessing import StandardScaler


def _safe_cov(X):
    """
    Return a covariance matrix with stable shape.
    - 1D input -> (1, 1)
    - Too few samples -> zeros
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        if X.shape[0] <= 1:
            return np.array([[0.0]], dtype=float)
        return np.array([[np.var(X, ddof=1)]], dtype=float)

    n_samples, n_features = X.shape
    if n_samples <= 1:
        return np.zeros((n_features, n_features), dtype=float)

    return np.atleast_2d(np.cov(X, rowvar=False, ddof=1))


def scatter_matrices(X, y):
    """
    Compute within-class and between-class scatter matrices:
        SW = sum_k (Nk / N) * Sigma_k
        SB = sum_k (Nk / N) * (mu_k - mu)(mu_k - mu)^T
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    classes = np.unique(y)
    N = len(y)
    d = X.shape[1]

    mu = X.mean(axis=0, keepdims=True)
    SW = np.zeros((d, d), dtype=float)
    SB = np.zeros((d, d), dtype=float)

    for cls in classes:
        Xk = X[y == cls]
        Nk = Xk.shape[0]

        mu_k = Xk.mean(axis=0, keepdims=True)
        Sigma_k = _safe_cov(Xk)

        SW += (Nk / N) * Sigma_k

        diff = mu_k - mu
        SB += (Nk / N) * (diff.T @ diff)

    return SW, SB


def scatter_criterion(X, y, eps=1e-12):
    """
    Lecture criterion:
        J = trace(SB) / trace(SW)
    """
    SW, SB = scatter_matrices(X, y)
    return float(np.trace(SB) / (np.trace(SW) + eps))


def forward_search(X, y, feature_names, candidate_feature_names, d=4, normalize=True):
    """
    Start from an empty set and add the feature that maximizes J
    at each step until d features are selected.
    """
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}
    candidate_indices = [feature_to_idx[name] for name in candidate_feature_names]
    Xcand = X[:, candidate_indices].astype(float)

    if normalize:
        Xcand = StandardScaler().fit_transform(Xcand)

    remaining = list(range(len(candidate_feature_names)))
    selected_local = []
    history = []

    for step in range(d):
        best_local = None
        best_j = -np.inf

        for local_idx in remaining:
            trial = selected_local + [local_idx]
            j_val = scatter_criterion(Xcand[:, trial], y)

            if j_val > best_j:
                best_j = j_val
                best_local = local_idx

        selected_local.append(best_local)
        remaining.remove(best_local)

        history.append({
            "step": step + 1,
            "added_feature": candidate_feature_names[best_local],
            "global_index": candidate_indices[best_local],
            "J": best_j,
            "current_set": [candidate_feature_names[i] for i in selected_local],
        })

    selected_global_indices = [candidate_indices[i] for i in selected_local]
    selected_names = [candidate_feature_names[i] for i in selected_local]

    return selected_global_indices, selected_names, history


def print_forward_history(history):
    print("\n===== FORWARD SEARCH USING SCATTER CRITERION =====")
    for item in history:
        current_set = ", ".join(item["current_set"])
        print(
            f"Step {item['step']}: add {item['added_feature']:22s} "
            f"-> J = {item['J']:.6f} | set = [{current_set}]"
        )
