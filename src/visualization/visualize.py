import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_labels):
    """
    Plots a confusion matrix using seaborn's heatmap.

    Parameters:
    y_true (list or array-like): True labels of the data.
    y_pred (list or array-like): Predicted labels of the data.
    class_labels (list): List of class labels to be displayed on the axes.

    Returns:
    None
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# You can add more visualization functions here