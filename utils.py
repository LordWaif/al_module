from typing import Callable, List, Union, Any
import numpy as np
from small_text import list_to_csr,TextDataset,csr_to_list
from scipy.sparse import csr_matrix
from dataset_class import DatasetLog

def transformY(records: list,function_label2int:Callable[[Any],List[int]],num_classes: int,isMultiLabel:bool, trainable_labels:list) -> Union[np.ndarray,csr_matrix]:
    function = lambda x: [trainable_labels.index(i) for i in x if i in trainable_labels]
    _y = [function([r for r in rec.annotation if r in trainable_labels]) for rec in records]
    if isMultiLabel:
        y = list_to_csr(_y,shape=(len(_y),num_classes))
    else:
        if len(trainable_labels) == 1:
            _y = np.array([1 if len(_y[i]) != 0 else 0 for i in range(len(_y))])
        y = _y
    return y

def createTD(trainable_labels: List[str], records: List[dict], dataset: DatasetLog, field: str = 'text') -> TextDataset:
    """
    Create a TextDataset object from a list of records and a DatasetLog object.

    Args:
        trainable_labels (List[str]): List of labels to train on.
        records (List[dict]): List of records to create the TextDataset from.
        dataset (DatasetLog): DatasetLog object containing label information.
        field (str, optional): Name of the field in the records to use as input. Defaults to 'text'.

    Returns:
        TextDataset: TextDataset object containing the inputs and labels.
    """
    y = transformY(records, dataset.LABEL2INT,
                   len(trainable_labels), dataset._multi_label, trainable_labels)
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
    
def label_encode(y):
    matriz_resultante = []
    for _y in y:
        nova_linha = [i for i, valor in enumerate(_y) if valor == 1]
        matriz_resultante.append(nova_linha)
    return np.array(matriz_resultante)



from small_text import PoolBasedActiveLearner  
from argilla import TextClassificationRecord  
from typing import List
from typing import List, Dict
import numpy as np

def predict(al: PoolBasedActiveLearner, data_json: Dict[str, any], data: List[TextClassificationRecord], dataset: DatasetLog) -> List[np.ndarray]:
    """
    Predicts the labels for a given set of data using a PoolBasedActiveLearner.

    Args:
        al (PoolBasedActiveLearner): The active learner used to predict the labels.
        data_json (Dict[str, any]): A dictionary containing the training labels, training field, and other relevant data.
        data (List[TextClassificationRecord]): A list of TextClassificationRecord objects containing the data to be predicted.
        dataset (DatasetLog): The dataset used for training the active learner.

    Returns:
        List[np.ndarray]: A list containing the predicted labels, true labels, and probability of each prediction.
    """
    records_textDataset: TextDataset = createTD(data_json['training_labels'],
                data, dataset, data_json['training_field'])
    csr_acc, proba_acc = al.classifier.predict(
                records_textDataset, return_proba=True
            )
    y_pred = hot_encode(csr_acc, len(data_json['training_labels']), dataset.multi_label)
    y_true = hot_encode(records_textDataset.y, len(data_json['training_labels']), dataset.multi_label)
    return y_true,y_pred,proba_acc