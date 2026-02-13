from preprocessing import load_and_preprocess_data
from models import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest
)
from evaluation import evaluate_model
from visualization import plot_predictions
from knowledge_extraction import plot_feature_importance
from rules import apply_rules
from hyperparameter_tuning import (
    tune_decision_tree,
    tune_random_forest
)

import pandas as pd

DATA_PATH = "data/california_housing.csv"


def main():
    # =====================================================
    # 1. Зареждане и предварителна обработка на данните
    # =====================================================
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(DATA_PATH)

    # =====================================================
    # 2. Базови модели
    # =====================================================
    lin_model = train_linear_regression(X_train, y_train)
    tree_model = train_decision_tree(X_train, y_train)
    forest_model = train_random_forest(X_train, y_train)

    # =====================================================
    # 3. Хиперпараметърна оптимизация
    # =====================================================
    print("\nХиперпараметърна оптимизация:")

    best_tree, tree_params, tree_score = tune_decision_tree(X_train, y_train)
    print("\nDecision Tree - най-добри параметри:")
    print(tree_params)
    print(f"Cross-validated R2: {tree_score:.4f}")

    best_forest, forest_params, forest_score = tune_random_forest(X_train, y_train)
    print("\nRandom Forest - най-добри параметри:")
    print(forest_params)
    print(f"Cross-validated R2: {forest_score:.4f}")

    # =====================================================
    # 4. Оценка и визуализация
    # =====================================================
    models = {
        "Linear Regression": lin_model,
        "Decision Tree": best_tree,
        "Random Forest": best_forest
    }

    for name, model in models.items():
        results = evaluate_model(model, X_test, y_test)

        print(f"\n{name}")
        print(f"MSE: {results['MSE']:.2f}")
        print(f"RMSE: {results['RMSE']:.2f}")
        print(f"R2: {results['R2']:.4f}")

        y_pred = model.predict(X_test)
        plot_predictions(y_test, y_pred, name)

    # =====================================================
    # 5. Извличане на знания (Feature Importance)
    # =====================================================
    feature_names = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"
    ]

    print("\nИзвличане на знания: важност на признаците")
    plot_feature_importance(best_forest, feature_names)

    # =====================================================
    # 6. Прогнозиране на цена на ново жилище
    # =====================================================
    new_house = pd.DataFrame([{
        "longitude": -118.29,
        "latitude": 34.05,
        "housing_median_age": 20,
        "total_rooms": 1800,
        "total_bedrooms": 300,
        "population": 800,
        "households": 280,
        "median_income": 6.5
    }])

    new_house_scaled = pd.DataFrame(
        scaler.transform(new_house),
        columns=new_house.columns
    )
    predicted_price = best_forest.predict(new_house_scaled)

    print("\nПрогнозна цена на жилището:")
    print(f"{predicted_price[0]:.2f} USD")

    # =====================================================
    # 7. Експертна интерпретация
    # =====================================================
    house_dict = {
        "longitude": -118.29,
        "latitude": 34.05,
        "housing_median_age": 20,
        "total_rooms": 1800,
        "total_bedrooms": 300,
        "population": 800,
        "households": 280,
        "median_income": 6.5
    }

    explanations = apply_rules(house_dict)

    print("\nЕкспертна интерпретация:")
    for exp in explanations:
        print("-", exp)


if __name__ == "__main__":
    main()
