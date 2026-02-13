import matplotlib.pyplot as plt


def plot_predictions(y_test, y_pred, title):
    """
    Визуализация: реални срещу прогнозирани стойности
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )
    plt.xlabel("Реални стойности")
    plt.ylabel("Прогнозирани стойности")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
