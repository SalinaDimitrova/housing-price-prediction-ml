import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(data_path):
    # =====================================================
    # 1. Зареждане на данните
    # =====================================================
    data = pd.read_csv(data_path)

    # =====================================================
    # 2. Основна информация за данните
    # =====================================================
    print("\nИнформация за набора от данни:")
    print(data.info())

    print("\nОписателна статистика:")
    print(data.describe())

    # =====================================================
    # 3. Хистограма на целевата променлива
    # =====================================================
    plt.figure(figsize=(8, 6))
    plt.hist(data["median_house_value"], bins=50)
    plt.xlabel("Цена на жилището")
    plt.ylabel("Брой наблюдения")
    plt.title("Разпределение на цените на жилищата")
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 4. Median Income vs House Value
    # =====================================================
    plt.figure(figsize=(8, 6))
    plt.scatter(
        data["median_income"],
        data["median_house_value"],
        alpha=0.3
    )
    plt.xlabel("Среден доход")
    plt.ylabel("Цена на жилището")
    plt.title("Зависимост между доход и цена")
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 5. Географско разпределение (Latitude / Longitude)
    # =====================================================
    plt.figure(figsize=(8, 6))
    plt.scatter(
        data["longitude"],
        data["latitude"],
        c=data["median_house_value"],
        cmap="viridis",
        alpha=0.5
    )
    plt.colorbar(label="Цена на жилището")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Географско разпределение на цените")
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 6. Корелационна матрица
    # =====================================================
    plt.figure(figsize=(10, 8))
    corr = data.select_dtypes(include=["float64", "int64"]).corr()

    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=False
    )
    plt.title("Корелационна матрица")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_eda("data/california_housing.csv")