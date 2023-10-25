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