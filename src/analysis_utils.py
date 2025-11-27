"""
Analysis helpers for dimensionality reduction and baseline classifiers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.plot_functions import plot_confusion_matrix

SEED = 42


# Used in: data_analysis.ipynb (sample down projections)
def subsample_for_viz(X, y, max_samples: int, random_state: int = SEED):
    """
    Subsample arrays for visualization without replacement.

    Args:
        X: Feature matrix.
        y: Labels array.
        max_samples: Maximum number of samples to keep.
        random_state: RNG seed.

    Returns:
        Tuple of (subsampled X, subsampled y, selected indices).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) <= max_samples:
        return X, y, np.arange(len(X))
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx], y[idx], idx


# Used in: data_analysis.ipynb (t-SNE configuration)
def resolve_tsne_perplexity(n_samples: int, requested: int | None = None) -> int:
    """
    Choose a valid t-SNE perplexity based on the sample count.

    Args:
        n_samples: Number of samples to embed.
        requested: Optional user-specified perplexity.

    Returns:
        Perplexity value within valid bounds.
    """
    if n_samples < 2:
        raise ValueError("t-SNE requires at least 2 samples.")
    if requested is not None:
        return min(requested, n_samples - 1)
    candidate = max(5, min(30, n_samples // 3 if n_samples > 9 else n_samples - 1))
    return min(candidate, n_samples - 1)


# Used in: data_analysis.ipynb (dimensionality reduction)
def run_dim_reduction(X, method: str = "pca", n_components: int = 2, scale: bool = True,
                      random_state: int = SEED, **kwargs):
    """
    Run PCA, MDS, or t-SNE on the provided data.

    Args:
        X: Input feature matrix.
        method: Reduction method ("pca", "mds", or "tsne").
        n_components: Number of output components.
        scale: Apply standard scaling before reduction.
        random_state: RNG seed.
        **kwargs: Additional parameters forwarded to the reducer.

    Returns:
        Tuple of (fitted model, coordinates).
    """
    method = method.lower()
    X_input = StandardScaler().fit_transform(X) if scale else X

    if method == "pca":
        model = PCA(n_components=n_components, random_state=random_state)
        coords = model.fit_transform(X_input)
    elif method == "mds":
        model = MDS(
            n_components=n_components,
            random_state=random_state,
            n_init=kwargs.pop("n_init", 4),
            max_iter=kwargs.pop("max_iter", 300),
            n_jobs=kwargs.pop("n_jobs", -1),
        )
        coords = model.fit_transform(X_input)
    elif method == "tsne":
        perplexity = resolve_tsne_perplexity(X_input.shape[0], kwargs.pop("perplexity", None))
        model = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perplexity,
            init=kwargs.pop("init", "pca"),
            learning_rate=kwargs.pop("learning_rate", "auto"),
            n_iter=kwargs.pop("n_iter", 1000),
            **kwargs,
        )
        coords = model.fit_transform(X_input)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")

    return model, coords


# Used in: data_analysis.ipynb (baseline embeddings evaluation)
def run_baseline_classifiers(X_train, y_train, X_test, y_test, label_order, random_state: int = SEED):
    """
    Train and evaluate simple baseline classifiers on embeddings.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        label_order: Ordered labels for reporting.
        random_state: RNG seed.

    Returns:
        DataFrame summarizing baseline accuracies.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, multi_class="auto", n_jobs=-1, random_state=random_state),
        "Perceptron": Perceptron(max_iter=500, random_state=random_state),
        "MLP (64,)": MLPClassifier(hidden_layer_sizes=(64,), max_iter=150, alpha=1e-4, random_state=random_state),
    }

    rows = []
    for name, clf in models.items():
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        acc = float((preds == y_test).mean())
        rows.append({"model": name, "accuracy": acc})
        print(f"=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, preds, labels=label_order))
        cm = confusion_matrix(y_test, preds, labels=label_order)
        plot_confusion_matrix(cm, label_order, title=f"{name} confusion matrix")
    return pd.DataFrame(rows)
