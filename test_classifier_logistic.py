import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load the test data
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')

    # Load the saved pipeline
    with open('optimized_logistic_regression_model.pkl', 'rb') as f:
        pipeline = pickle.load(f)

    # Use the pipeline to make predictions on the test data
    y_pred = pipeline.predict(X_test)

    # Get the predicted probabilities for each class
    y_pred_proba = pipeline.predict_proba(X_test)

    # The second column represents the probability of heart disease presence
    heart_disease_probabilities = y_pred_proba[:, 1]

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Patient ID': range(1, len(X_test) + 1),
        'Predicted Outcome': y_pred,
        'Actual Outcome': y_test['target'],
        'Heart Disease Probability': heart_disease_probabilities
    })

    # Print the results for each patient
    print("Patient ID\tPredicted Outcome\tActual Outcome\tHeart Disease Probability")
    for index, row in results_df.iterrows():
        print(f"{row['Patient ID']}\t\t{row['Predicted Outcome']}\t\t\t{row['Actual Outcome']}\t\t{row['Heart Disease Probability']:.2f}")

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nTest set results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")

    # Plot the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    plt.xlabel('Predicted Outcome')
    plt.ylabel('Actual Outcome')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    main()
