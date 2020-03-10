import matplotlib.pyplot as plt
import os
import numpy as np


def plot_coef(title, coef, feature_names, top_features=20):
    """Create feature importance plot."""
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack(
        [top_negative_coefficients, top_positive_coefficients])
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features),
               feature_names[top_coefficients], rotation=60, ha='right')
    plt.title(title)
    os.makedirs('feature_plots', exist_ok=True)
    plt.savefig(f'feature_plots/{title}.pdf', bbox_inches="tight")
    plt.close()
