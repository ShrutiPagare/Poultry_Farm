"""Shared evaluation & plotting helpers."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

PALETTE = "#2563EB"

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path: Path = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(model, feature_names, title, save_path: Path = None, top_n=15):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    idx  = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in idx][::-1],
            importances[idx][::-1], color=PALETTE)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_regression_scatter(y_true, y_pred, title, save_path: Path = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, color=PALETTE, s=15)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=2)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
