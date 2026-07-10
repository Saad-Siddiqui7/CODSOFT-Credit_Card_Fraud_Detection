from sklearn.metrics import accuracy_score, confusion_matrix,classification_report


def evaluate_model(y_test,y_pred):

    print("Accuracy:",accuracy_score(y_test,y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))