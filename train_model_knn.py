import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load the dataset
data = pd.read_csv('heart.csv')

# Split the data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)


# Create a pipeline for KNN
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

# Define the hyperparameter grid for KNN
knn_param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__p': [1, 2]
}

# Perform grid search with cross-validation for KNN
knn_grid_search = GridSearchCV(knn_pipeline, knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
knn_grid_search.fit(X_train, y_train)

# Get the best models

knn_best_model = knn_grid_search.best_estimator_





print(f"Best cross-validation accuracy: {knn_grid_search.best_score_:.3f}")
print(f"Best parameters found: {knn_grid_search.best_params_}")
print("Best Hyperparameters: {}".format(knn_grid_search.best_params_))
# Save the optimized model
with open('optimized_knn_model.pkl', 'wb') as file:
    pickle.dump(knn_grid_search.best_estimator_, file)
