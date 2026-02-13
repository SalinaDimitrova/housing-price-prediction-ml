from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def tune_decision_tree(X_train, y_train):
    param_grid = {
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10]
    }

    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 15, None],
        "min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_
