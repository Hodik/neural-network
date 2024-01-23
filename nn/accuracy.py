import numpy as np


def calculate_accuracy(pred, y, tolerance=0.5):
    """
    Calculate custom accuracy based on Mean Squared Error.

    Parameters:
    - y_true: numpy array, true labels (ground truth)
    - y_pred: numpy array, predicted values
    - tolerance: float, acceptable tolerance for correct prediction

    Returns:
    - accuracy: float, custom accuracy
    """
    accuracy = np.mean((np.abs(y - pred) < tolerance).astype(float))
    return accuracy


def calculate_accuracy_with_probabilities_one_hot(pred, y):
    """
    Calculate accuracy given true one-hot encoded labels and predicted probabilities.

    Parameters:
    - y_true: numpy array, true one-hot encoded labels (ground truth)
    - y_pred_probabilities: numpy array, predicted probabilities

    Returns:
    - accuracy: float, accuracy of predictions
    """
    # Convert predicted probabilities to predicted one-hot encoded labels
    predicted_labels_one_hot = (pred == pred.max(axis=1, keepdims=True)).astype(int)

    # Compare predicted labels with true labels
    correct_predictions = np.sum(np.all(predicted_labels_one_hot == y, axis=1))

    # Calculate accuracy
    accuracy = correct_predictions / y.shape[0]

    return accuracy
