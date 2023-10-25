from sklearn.metrics import accuracy_score, f1_score,recall_score,precision_score,hamming_loss
import numpy as np

def calcule_metrics(metricas:dict,y_true:np.array,y_pred:np.array,proba:np.array,batch:int):
    metricas['acc'].append(accuracy_score(y_true,y_pred))
    metricas['f1'].append(f1_score(y_true,y_pred,average='weighted'))
    metricas['hs'].append(hamming_loss(y_true,y_pred))
    metricas['precision'].append(precision_score(y_true,y_pred,average='weighted'))
    metricas['recall'].append(recall_score(y_true,y_pred,average='weighted'))
    metricas['batch'].append(batch)
    metricas['confidence'].append(np.mean(proba))
    return metricas