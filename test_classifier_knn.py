import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from sklearn.model_selection import train_test_split

# Load the testing data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# Load the trained model
with open('optimized_knn_model.pkl', 'rb') as file:
    knn_best_model = pickle.load(file)

# Make predictions on the test set

knn_y_pred = knn_best_model.predict(X_test)



# Calculate evaluation metrics for KNN
knn_accuracy = accuracy_score(y_test, knn_y_pred)
knn_precision = precision_score(y_test, knn_y_pred)
knn_recall = recall_score(y_test, knn_y_pred)
knn_f1 = f1_score(y_test, knn_y_pred)


results_df = pd.DataFrame({'Patient ID': range(1, len(X_test) + 1),
                           'Predicted Outcome': knn_y_pred,
                           'Actual Outcome': y_test['target']})

# Print the predicted outcome and actual outcome for each patient
print("Patient ID\tPredicted Outcome\tActual Outcome")
for index, row in results_df.iterrows():
    print(f"{row['Patient ID']}\t\t{row['Predicted Outcome']}\t\t\t{row['Actual Outcome']}")





print("\nK-Nearest Neighbors:")
print("Accuracy: {:.3f}".format(knn_accuracy))
print("Precision: {:.3f}".format(knn_precision))
print("Recall: {:.3f}".format(knn_recall))
print("F1-score: {:.3f}".format(knn_f1))
