import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the testing data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# Load the trained model
with open('optimized_RandomForest_regression_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Create a DataFrame to store the results
results_df = pd.DataFrame({'Patient ID': range(1, len(X_test) + 1),
                           'Predicted Outcome': y_pred,
                           'Actual Outcome': y_test['target']})

# Print the predicted outcome and actual outcome for each patient
print("Patient ID\tPredicted Outcome\tActual Outcome")
for index, row in results_df.iterrows():
    print(f"{row['Patient ID']}\t\t{row['Predicted Outcome']}\t\t\t{row['Actual Outcome']}")

# Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nTest set results:")
print("Accuracy: {:.3f}".format(accuracy))
print("Precision: {:.3f}".format(precision))
print("Recall: {:.3f}".format(recall))
print("F1-score: {:.3f}".format(f1))


# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted Outcome')
plt.ylabel('Actual Outcome')
plt.title('Confusion Matrix for KNN Model')
plt.show()
