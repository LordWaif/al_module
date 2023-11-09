import pandas as pd
import random as rd

def generateArtificialDataset(len_text:int=12,frac_isSend:float=0.3,quantity:int=100,multi_label:bool=True,num_labels:int=2) -> pd.DataFrame:
    """
    Generates an artificial dataset with random text and labels.

    Args:
    - len_text (int): length of the random text to be generated.
    - frac_isSend (float): fraction of the dataset that will be used for pre-training.
    - quantity (int): number of samples to be generated.
    - multi_label (bool): whether to allow multiple labels per sample or not.
    - num_labels (int): number of labels to be generated.

    Returns:
    - pd.DataFrame: a pandas dataframe containing the generated samples and labels.
    """
    labels = {f'label{_n+1}':0 for _n in range(num_labels)}
    lines = []
    for i in range(quantity):
        line = {'text':''.join(map(chr,rd.sample(range(65,90),len_text))),'_isSend':True}
        line.update(labels)
        if not multi_label:
            key_true = rd.sample(list(labels.keys()),1)[0]
            line[key_true] = 1
        else:
            for i in labels.keys():
                line[i] = rd.randint(0,1)
        lines.append(line)
    
    df = pd.DataFrame(lines)
    preTraining = df.sample(frac=frac_isSend)
    df.loc[preTraining.index,df.columns[1:]] = 0
    df.loc[preTraining.index,'_isSend'] = False
    return df