import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')


with open('optimized_knn_model.pkl', 'rb') as file:
    knn_best_model = pickle.load(file)


knn_y_pred = knn_best_model.predict(X_test)



knn_accuracy = accuracy_score(y_test, knn_y_pred)
knn_precision = precision_score(y_test, knn_y_pred)
knn_recall = recall_score(y_test, knn_y_pred)
knn_f1 = f1_score(y_test, knn_y_pred)


results_df = pd.DataFrame({'Patient ID': range(1, len(X_test) + 1),
                           'Predicted Outcome': knn_y_pred,
                           'Actual Outcome': y_test['target']})


print("Patient ID\tPredicted Outcome\tActual Outcome")
for index, row in results_df.iterrows():
    print(f"{row['Patient ID']}\t\t{row['Predicted Outcome']}\t\t\t{row['Actual Outcome']}")





print("\nK-Nearest Neighbors:")
print("Accuracy: {:.3f}".format(knn_accuracy))
print("Precision: {:.3f}".format(knn_precision))
print("Recall: {:.3f}".format(knn_recall))
print("F1-score: {:.3f}".format(knn_f1))




conf_matrix = confusion_matrix(y_test, knn_y_pred)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted Outcome')
plt.ylabel('Actual Outcome')
plt.title('Confusion Matrix for KNN Model')
plt.show()
