from small_text.integrations.transformers.classifiers.factories import (
    SetFitClassificationFactory
)
from small_text.integrations.transformers.classifiers.setfit import SetFitModelArguments
from small_text import PoolBasedActiveLearner, SubsamplingQueryStrategy
from small_text.query_strategies import QueryStrategy
from dataset_class import DatasetLog

def factory(
    model_name: str,  # The name of the sentence transformer model to be used for classification
    num_classes: int,  # The number of classes in the classification task
    epochs: int,  # The number of epochs to train the classification model
    multi_label: bool,  # Whether the classification task is multi-label or not
    batch: int,  # The batch size to use during training
)-> SetFitClassificationFactory:
    """
    Factory function to create a SetFitClassificationFactory object for a given sentence transformer model.

    Args:
        model_name (str): The name of the sentence transformer model to be used for classification.
        num_classes (int): The number of classes in the classification task.
        epochs (int): The number of epochs to train the classification model.
        multi_label (bool): Whether the classification task is multi-label or not.
        batch (int): The batch size to use during training.

    Returns:
        SetFitClassificationFactory: A factory object that can be used to create a classification model.
    """
    sentence_transformer_model_name = model_name
    setfit_model_args = SetFitModelArguments(sentence_transformer_model_name)
    clf_factory = SetFitClassificationFactory(
        setfit_model_args,
        num_classes,
        classification_kwargs={
            "trainer_kwargs": {"num_epochs": epochs},
            "multi_label": multi_label,
            "mini_batch_size": batch,
        },
    )
    return clf_factory

def initialize_activeLearning(
    query: QueryStrategy,  # The query strategy to be used for active learning
    dataset: DatasetLog,  # The dataset to be used for active learning
    factory: SetFitClassificationFactory,  # The classification factory to be used for active learning
)-> PoolBasedActiveLearner:
    """
    Initializes an active learning process with the given query strategy, dataset, and classification factory.

    Args:
        query (QueryStrategy): The query strategy to be used for active learning.
        dataset (DatasetLog): The dataset to be used for active learning.
        factory (SetFitClassificationFactory): The classification factory to be used for active learning.

    Returns:
        PoolBasedActiveLearner: An active learner object initialized with the given parameters.
    """
    query_strategy = SubsamplingQueryStrategy(query)
    active_learner = PoolBasedActiveLearner(
        factory, query_strategy, dataset.textDataset(False)
    )
    return active_learner
