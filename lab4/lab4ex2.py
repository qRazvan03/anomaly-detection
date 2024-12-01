import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.datasets import fetch_openml

cardio = fetch_openml(name='cardiotocography', version=1, as_frame=False)
X, y = cardio.data, cardio.target

y = np.where(y == '1', 1, -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

parameter_grid = {
    'ocsvm__kernel': ['linear', 'rbf', 'poly'],
    'ocsvm__gamma': ['scale', 0.1, 0.5, 1],
    'ocsvm__nu': [0.05, 0.1, 0.15]
}

pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('ocsvm', OneClassSVM())  # One-class SVM model
])

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring=make_scorer(balanced_accuracy_score, greater_is_better=True),
    cv=3,  # 3-fold cross-validation
    verbose=3
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
y_test_pred = best_model.predict(X_test)

y_test_pred_pyod = (-1 * y_test_pred + 1) // 2

ba_test = balanced_accuracy_score(y_test, y_test_pred)

print("Best parameters found by GridSearchCV:", best_params)
print("Balanced Accuracy on the test set:", ba_test)