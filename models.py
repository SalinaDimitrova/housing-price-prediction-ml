from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, max_depth=10):
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=15):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model
