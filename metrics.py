import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score

def calcule_metrics(metricas:dict,y_true:np.array,y_pred:np.array,proba:np.array,batch:int):
    """
    Calculates various metrics for a given set of true and predicted labels and probabilities.

    Args:
    metricas (dict): A dictionary containing the metrics to be calculated.
    y_true (np.array): An array of true labels.
    y_pred (np.array): An array of predicted labels.
    proba (np.array): An array of predicted probabilities.
    batch (int): The batch number.

    Returns:
    dict: A dictionary containing the updated metrics.
    """
    metricas['acc'].append(accuracy_score(y_true,y_pred))
    metricas['f1'].append(f1_score(y_true,y_pred,average='weighted'))
    metricas['hs'].append(hamming_loss(y_true,y_pred))
    metricas['precision'].append(precision_score(y_true,y_pred,average='weighted'))
    metricas['recall'].append(recall_score(y_true,y_pred,average='weighted'))
    metricas['batch'].append(batch)
    metricas['confidence'].append(np.mean(proba))
    return metricas

# target_metrics = ['acc','f1','hs','precision','recall']
def earlyStopping(target_metric: str, metrics: dict, patience: int = float('inf'), threshold: float = float('inf')) -> bool:
    """
    Determines whether to stop training early based on the given target metric and stopping criteria.

    Args:
        target_metric (str): The metric to monitor for early stopping.
        metrics (dict): A dictionary containing the history of all metrics.
        patience (int, optional): The number of epochs to wait before stopping if the target metric does not improve. Defaults to infinity.
        threshold (float, optional): The threshold value for the target metric. If the target metric exceeds this value, training will stop. Defaults to infinity.

    Returns:
        bool: True if training should stop, False otherwise.
    """
    if len(metrics[target_metric]) == 0:
        return False
    if target_metric != 'h1':
        if metrics[target_metric][-1] >= threshold or len(metrics[target_metric])-metrics[target_metric].index(max(metrics[target_metric]))> patience:
            return True
    else:
        if metrics[target_metric][-1] <= threshold or len(metrics[target_metric])-metrics[target_metric].index(min(metrics[target_metric]))> patience:
            return True
    return False