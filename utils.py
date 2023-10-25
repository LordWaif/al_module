from typing import Callable, List, Union, Any
import numpy as np
from small_text import list_to_csr,TextDataset,csr_to_list
from scipy.sparse import csr_matrix
from dataset_class import DatasetLog

def transformY(records: list,function_label2int:Callable[[Any],List[int]],num_classes: int,isMultiLabel:bool, trainable_labels:list) -> Union[np.ndarray,csr_matrix]:
    _y = np.array([function_label2int([r for r in rec.annotation if r in trainable_labels]) for rec in records],dtype=object)
    if isMultiLabel:
        y = list_to_csr(_y, shape=(len(_y), num_classes))
    else:
        if len(trainable_labels) == 1:
            _y = np.array([1 if len(_y[i]) != 0 else 0 for i in range(len(_y))])
        y = _y
    return y

def createTD(trainable_labels,records:list,dataset:DatasetLog,field:str='text') -> TextDataset:
    y = transformY(records, dataset.LABEL2INT,
                   len(trainable_labels), dataset._multi_label,trainable_labels)
    td = TextDataset(
        [_.inputs[field] for _ in records],
        y,
        target_labels=np.arange(2 if len(trainable_labels) == 1 else len(trainable_labels))
    )
    return td

def _hot_encode(y, n_classes:int, isMultiLabel:bool):
    if n_classes == 1:
        return y
    _zeros = np.zeros(shape=(len(y), n_classes), dtype=np.int8)
    for i, _y in enumerate(y):
        if isMultiLabel:
            for j in _y:
                _zeros[i][j] = 1
        else:
            for j in _y:
                _zeros[i][j] = 1
    return _zeros


def hot_encode(y:Union[np.ndarray,csr_matrix], n_classes:int, isMultiLabel:bool):
    if isMultiLabel:
        return _hot_encode(csr_to_list(y), n_classes, isMultiLabel)
    else:
        return _hot_encode(y, n_classes, isMultiLabel)

from small_text import PoolBasedActiveLearner  
from argilla import TextClassificationRecord  
from typing import List
def predict(al:PoolBasedActiveLearner,data_json:dict,data:List[TextClassificationRecord],dataset:DatasetLog) -> List[np.ndarray]:
    records_textDataset: TextDataset = createTD(data_json['training_labels'],
                data, dataset, data_json['training_field'])
    csr_acc, proba_acc = al.classifier.predict(
                records_textDataset, return_proba=True
            )
    y_pred = hot_encode(csr_acc, len(data_json['training_labels']), dataset.multi_label)
    y_true = hot_encode(records_textDataset.y, len(data_json['training_labels']), dataset.multi_label)
    return y_true,y_pred,proba_acc