import numpy as np


def POD(TP, FN):
    """
    Compute the Probability of Detection (POD).
    
    Args:
        TP (int): number of true positives.
        FN (int): number of false negatives.
    
    Returns:
        float: Probability of Detection.
    """
    return TP / (TP + FN)

def FAR(FP, TP):
    """
    Compute the False Alarm Rate (FAR).
    
    Args:
        FP (int): number of false positives.
        TP (int): number of true positives.
    
    Returns:
        float: False Alarm Rate.
    """
    return FP / (FP + TP)

def bias(TP, FP, FN):
    """
    Compute the bias.
    
    Args:
        TP (int): number of true positives.
        FP (int): number of false positives.
        FN (int): number of false negatives.
    
    Returns:
        float: bias.
    """
    return (TP + FP) / (TP + FN)

def HSS(TP, FP, FN, TN):
    """
    Compute the Heidke Skill Score (HSS).
    
    Args:
        TP (int): number of true positives.
        FP (int): number of false positives.
        FN (int): number of false negatives.
        TN (int): number of true negatives.
    
    Returns:
        float: Heidke Skill Score.
    """
    return (2 * (TP * TN - FP * FN)) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))

def ETS(TP, FP, FN, TN):
    """
    Compute the Equitable Threat Score (ETS).
    
    Args:
        TP (int): number of true positives.
        FP (int): number of false positives.
        FN (int): number of false negatives.
        TN (int): number of true negatives.
    
    Returns:
        float: Equitable Threat Score.
    """
    TP_ref = (TP + FN) * (TP + FP) / (TP + FN + FP + TN)
    return (TP - TP_ref) / (TP + FP + FN - TP_ref)

def CSI(TP, FP, FN):
    """
    Compute the Critical Success Index (CSI).
    
    Args:
        TP (int): number of true positives.
        FP (int): number of false positives.
        FN (int): number of false negatives.
    
    Returns:
        float: Critical Success Index.
    """
    return TP / (TP + FP + FN)

def TSS(TP, FP, FN, TN):
    """
    Compute the True Skill Score (TSS).
    
    Args:
        TP (int): number of true positives.
        FP (int): number of false positives.
        FN (int): number of false negatives.
        TN (int): number of true negatives.
    
    Returns:
        float: True Skill Score.
    """
    return TP / (TP + FN) - FP / (FP + TN)

def MCC(TP, FP, FN, TN):
    """
    Compute the Matthews Correlation Coefficient (MCC).
    
    Args:
        TP (int): number of true positives.
        FP (int): number of false positives.
        FN (int): number of false negatives.
        TN (int): number of true negatives.
    
    Returns:
        float: Matthews Correlation Coefficient.
    """
    return (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

def F1(TP, FP, FN):
    """
    Compute the F1 score.
    
    Args:
        TP (int): number of true positives.
        FP (int): number of false positives.
        FN (int): number of false negatives.
    
    Returns:
        float: F1 score.
    """
    return 2 * TP / (2 * TP + FP + FN)

def precision(TP, FP):
    """
    Compute the precision.
    
    Args:
        TP (int): number of true positives.
        FP (int): number of false positives.
    
    Returns:
        float: precision.
    """
    return TP / (TP + FP)

def recall(TP, FN):
    """
    Compute the recall.
    
    Args:
        TP (int): number of true positives.
        FN (int): number of false negatives.
    
    Returns:
        float: recall.
    """
    return TP / (TP + FN)

def binary_cross_entropy(y_true, y_pred):
    """
    Compute Binary Cross Entropy (BCE) loss.

    Parameters:
        y_true (np.array): Flattened array of true binary labels (0 or 1).
        y_pred (np.array): Flattened array of predicted probabilities (range: 0 to 1).
    
    Returns:
        float: The BCE loss averaged over all samples.
    """
    # Ensure the arrays are floating point
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Add a small constant to avoid log(0) (numerical stability)
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # Compute BCE as the negative average of 
    # [y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce

def mean_squared_error(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE).

    Parameters:
        y_true (np.array): Flattened array of true binary values (0 or 1).
        y_pred (np.array): Flattened array of predicted continuous values.
    
    Returns:
        float: The average squared difference between true and predicted values.
    """
    # Ensure the arrays are floating point
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def mean_absolute_error(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE).

    Parameters:
        y_true (np.array): Flattened array of true binary values (0 or 1).
        y_pred (np.array): Flattened array of predicted continuous values.
    
    Returns:
        float: The average absolute difference between true and predicted values.
    """
    # Ensure the arrays are floating point
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def confusion_matrix(y_true, y_pred, num_classes, plot=False):
    """
    Compute the confusion matrix for a multi-class classification problem.
    
    Args:
        y_true (np.ndarray): true labels, shape (n_samples,).
        y_pred (np.ndarray): predicted labels, shape (n_samples,).
        num_classes (int): number of classes.
    
    Returns:
        int a: number of true positives.
        int b: number of false positives.
        int c: number of false negatives.
        int d: number of true negatives.
    """
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    # Fill confusion matrix
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = np.sum((y_true == i) & (y_pred == j))
    
    # Compute TP, FP, FN, TN
    TP = np.diag(confusion_matrix)
    FP = np.sum(confusion_matrix, axis=0) - TP
    FN = np.sum(confusion_matrix, axis=1) - TP
    TN = np.sum(confusion_matrix) - (TP + FP + FN)
    
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        class_labels = ['Stratiform', 'Convective'] if num_classes == 2 else [str(i) for i in range(num_classes)]

        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    return TP, FP, FN, TN

def compute_metrics(y_true, y_pred, num_classes=2):
    """
    Compute evaluation metrics for a multi-class classification problem.
    
    Args:
        y_true (np.ndarray): true labels, shape (n_samples,).
        y_pred (np.ndarray): predicted labels, shape (n_samples,).
        num_classes (int): number of classes.
    
    Returns:
        dict: evaluation metrics.
    """
    
    # Round all the predicted values to the nearest integer
    y_pred = np.round(y_pred).astype(int)
    
    
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred, num_classes)
    
    metrics = {}
    metrics["POD"] = POD(TP, FN)
    metrics["FAR"] = FAR(FP, TP)
    metrics["bias"] = bias(TP, FP, FN)
    metrics["HSS"] = HSS(TP, FP, FN, TN)
    metrics["ETS"] = ETS(TP, FP, FN, TN)
    metrics["CSI"] = CSI(TP, FP, FN)
    metrics["TSS"] = TSS(TP, FP, FN, TN)
    metrics["MCC"] = MCC(TP, FP, FN, TN)
    metrics["F1"] = F1(TP, FP, FN)
    metrics["precision"] = precision(TP, FP)
    metrics["recall"] = recall(TP, FN)
    
    #if any of the metrics are NaN, return None
    for metric in metrics:
        # Check if the metric value is an array or scalar
        metric_value = metrics[metric]
        # Use np.any() to check for NaN if it's an array, otherwise check directly
        if isinstance(metric_value, np.ndarray):
            if np.any(np.isnan(metric_value)):
                print(f"Warning: NaN found in metric '{metric}'. Returning None.")
                return None
        elif np.isnan(metric_value):
             print(f"Warning: NaN found in metric '{metric}'. Returning None.")
             return None

    return metrics

def print_metrics(metrics):
    """
    Print evaluation metrics.
    
    Args:
        metrics (dict): evaluation metrics.
    """
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

def metrics_to_string(metrics):
    """
    Convert evaluation metrics to a string.
    
    Args:
        metrics (dict): evaluation metrics.
    
    Returns:
        str: formatted string of evaluation metrics.
    """
    metrics_str = ""
    for metric, value in metrics.items():
        metrics_str += f"{metric}: {value}\n"
    return metrics_str