import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import pickle


data_path = 'heart.csv'
heart_data = pd.read_csv(data_path)


X = heart_data.drop('target', axis=1)
y = heart_data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)


categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_features = [col for col in X.columns if col not in categorical_features]


numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])


categorical_transformer = OneHotEncoder(handle_unknown='ignore')


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


pca = PCA()


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', pca),
    ('feature_selection', SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", random_state=42))),
    ('classifier', LogisticRegression(max_iter=10000, random_state=42))
])


param_grid = {
    'pca__n_components': [None, 5, 10, 15],  # None means all components will be used
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l2'],  # 'l1' can be explored with 'liblinear' solver
    'feature_selection__estimator__C': [0.01, 0.1, 1, 10, 100]
}


grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best cross-validation accuracy: {grid_search.best_score_:.3f}")
print(f"Best parameters found: {grid_search.best_params_}")


with open('optimized_logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(grid_search.best_estimator_, file)
