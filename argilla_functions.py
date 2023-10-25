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

def initialize_log(dataset: DatasetLog, rg: argilla, workspace: str, inputs: dict) -> None:
    print(dataset.LABELS+['__control__'])
    settings = rg.TextClassificationSettings(label_schema=dataset.LABELS+['__control__'])
    try:
        rg.delete_dataset(name=dataset.name, workspace=workspace)
    except:
        pass
    rg.active_client().set_workspace(workspace)
    rg.configure_dataset_settings(
        name=dataset.name, settings=settings, workspace=workspace)
    # Create the initial batch
    initial_indices = dataset.random_sample()
    dataset._dataset.loc[initial_indices, "_isSend"] = True
    records = createRecords(dataset, 0, initial_indices, rg, inputs)
    #print(records[0])
    rg.log(records=records, name=dataset.name,
           workspace=workspace)