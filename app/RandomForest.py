import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv('app/data/World Bank Dataset.csv')

# Assuming 'GDP' is the target variable and the rest are features
X = data.drop(['Country Name', 'Year', 'GDP (current US$)'], axis=1)
y = data['GDP (current US$)']

# Print the number of features
num_features = X.shape[1]  # Number of columns in X
print(f'Number of features in the input of the model: {num_features}')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100],
    'max_features': ['sqrt'],  # Removed 'auto'
    'max_depth': [20],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'monotonic_cst': [None]
}

# Setup the GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best estimator
best_model = grid_search.best_estimator_

# Save the best model
joblib.dump(best_model, 'optimized_random_forest_gdp_model.pkl')

# Predict on the test set
y_pred = best_model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2) Score: {r2}')