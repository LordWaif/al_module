import pandas as pd

import pandas as pd
import numpy as np
import pickle
from small_text import TextDataset, list_to_csr

def createTextDataset(path_test:str,training_labels:list,multi_label:bool) -> None:
    """
    Creates a text dataset from a CSV file and saves it as a pickle file.

    Args:
    - path_test (str): The path to the CSV file.
    - training_labels (list): A list of labels for the training data.
    - multi_label (bool): A boolean indicating whether the dataset has multiple labels.

    Returns:
    - None
    """
    test = pd.read_csv(path_test)
    x = test['text'].values
    
    def Int2Label(x, columns): 
        return [columns.index(lb) for i, lb in zip(
            x[columns].values, columns) if i == 1]
    
    if multi_label:
        y = []
        for i in test.index:
            y.append(Int2Label(test.loc[i],training_labels))
        y = np.array(y)
        y = list_to_csr(y,shape=(len(y),len(training_labels)))
    else:
        y = test[training_labels].values
    
    teste_tds = TextDataset(x,y,target_labels=np.arange(2 if len(training_labels)== 1 else len(training_labels)))
    
    file = open(path_test.replace('.csv','.pkl'),'wb')
    pickle.dump(teste_tds,file)
    file.close()
    