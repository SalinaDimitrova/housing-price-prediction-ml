import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(path):
    # =====================================================
    # 1. Зареждане на данните
    # =====================================================
    data = pd.read_csv(path)

    # =====================================================
    # 2. Премахване на неинформативни колони (ако има)
    # =====================================================
    if "id" in data.columns:
        data = data.drop("id", axis=1)

    # =====================================================
    # 3. Обработка на липсващи стойности
    # Kaggle dataset има NaN в total_bedrooms
    # =====================================================
    if "total_bedrooms" in data.columns:
        data["total_bedrooms"].fillna(
            data["total_bedrooms"].median(),
            inplace=True
        )

    # =====================================================
    # 4. Дефиниране на целева и входни променливи
    # =====================================================
    y = data["median_house_value"]
    X = data.drop(
        ["median_house_value", "ocean_proximity"],
        axis=1
    )

    # =====================================================
    # 5. Разделяне на train/test
    # =====================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =====================================================
    # 6. Стандартизация на числовите признаци
    # =====================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
