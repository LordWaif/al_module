from dataset_class import DatasetLog
import argilla
import numpy as np
from typing import List
from argilla import TextClassificationRecord

def create_user(rg: argilla, **kwargs):
    try:
        user = rg.User.create(**kwargs)
        work = rg.Workspace.create(kwargs.get('username'))
        work.add_user(user.id)
    except ValueError:
        pass

def createRecords(
    datasets: DatasetLog,
    batch: int,
    indices: np.ndarray,
    rg: argilla,
    input: dict,
) -> List[TextClassificationRecord]:
    """
    Creates a list of TextClassificationRecord objects based on the given dataset and input parameters.

    Args:
        datasets (DatasetLog): The dataset to create records from.
        batch (int): The batch ID to assign to the new records.
        indices (np.ndarray): The indices of the dataset to create records from.
        rg (argilla): The argilla object to use for creating the new records.
        input (dict): A dictionary mapping input keys to column names in the dataset.

    Returns:
        List[TextClassificationRecord]: A list of TextClassificationRecord objects created from the given dataset and input parameters.
    """
    def Int2Label(x, columns): return [lb for i, lb in zip(
        x[columns].values, columns) if i == 1]
    new_records = []
    for idx in indices:
        rotulos = Int2Label(datasets.dataset.iloc[idx], datasets.LABELS)
        if len(rotulos) == 0:
            status = 'Default'
        else:
            status = 'Validated'
        new_records.append(
            rg.TextClassificationRecord(
                inputs={
                    k: str(datasets.dataset.iloc[idx][input[k]]) for k in input.keys()},
                metadata={"batch_id": batch},
                id=idx,
                annotation=['__control__']+rotulos,
                multi_label=True,
                status=status,
            )
        )
    return new_records

def initialize_log(dataset: DatasetLog, rg: argilla, workspace: str, inputs: dict,isAl:bool) -> None:
    """
    Initializes the logging process for the given dataset using the specified argilla client and workspace.

    Args:
        dataset (DatasetLog): The dataset to be logged.
        rg (argilla): The argilla client to be used for logging.
        workspace (str): The name of the workspace to be used for logging.
        inputs (dict): A dictionary containing the input data for the logging process.

    Returns:
        None
    """
    settings = rg.TextClassificationSettings(label_schema=dataset.LABELS+['__control__'])
    try:
        rg.delete_dataset(name=dataset.name, workspace=workspace)
    except:
        pass
    rg.active_client().set_workspace(workspace)
    rg.configure_dataset_settings(
        name=dataset.name, settings=settings, workspace=workspace)
    # Create the initial batch
    initial_indices = dataset.random_sample(isAl)
    dataset._dataset.loc[initial_indices, "_isSend"] = True
    records = createRecords(dataset, 0, initial_indices, rg, inputs)
    #print(records[0])
    rg.log(records=records, name=dataset.name,
           workspace=workspace)