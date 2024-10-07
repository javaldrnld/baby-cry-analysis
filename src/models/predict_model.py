from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate the performance of a given model on test data.
    Parameters:
    model (sklearn.base.BaseEstimator): The trained model to evaluate.
    X_test (numpy.ndarray or pandas.DataFrame): The test features.
    y_test (numpy.ndarray or pandas.Series): The true labels for the test data.
    scaler (sklearn.preprocessing.StandardScaler or similar): The scaler used to transform the test features.
    Returns:
    tuple: A tuple containing the following evaluation metrics:
        - accuracy (float): The accuracy of the model.
        - precision (float): The precision of the model, calculated with macro averaging.
        - recall (float): The recall of the model, calculated with macro averaging.
        - f1 (float): The F1 score of the model, calculated with macro averaging.
        - classification_report (str): A text summary of the precision, recall, and F1 score for each class.
    """
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    return accuracy, precision, recall, f1, classification_report(y_test, y_pred, zero_division=0)

# You can add more model evaluation and prediction functions here