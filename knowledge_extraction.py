import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importance(model, feature_names):
    """
    Визуализация на важността на признаците
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(
        range(len(importances)),
        importances[indices],
        align="center"
    )
    plt.xticks(
        range(len(importances)),
        [feature_names[i] for i in indices],
        rotation=45
    )
    plt.title("Важност на признаците (Feature Importance)")
    plt.ylabel("Относителна важност")
    plt.tight_layout()
    plt.show()

    return importances
