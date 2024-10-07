from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def preprocess_data(df):
    """
    Preprocesses the input DataFrame by encoding the labels and splitting the data into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and a 'label' column.

    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray): Training features.
            - X_test (np.ndarray): Testing features.
            - y_train (np.ndarray): Training labels.
            - y_test (np.ndarray): Testing labels.
            - label_encoder (LabelEncoder): The fitted label encoder.
    """

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return train_test_split(X, y, test_size=0.2, random_state=42), label_encoder

def train_models(X_train, y_train):
    """
    Trains multiple machine learning models on the provided training data.
    Parameters:
    X_train (array-like): The training input samples.
    y_train (array-like): The target values (class labels) as integers or strings.
    Returns:
    tuple: A tuple containing:
        - models (dict): A dictionary where keys are model names and values are the trained model instances.
        - scaler (StandardScaler): The fitted StandardScaler instance used to scale the training data.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    models = {
        'SVM': SVC(kernel='rbf', C=12, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=32, criterion='entropy', random_state=42),
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models, scaler
