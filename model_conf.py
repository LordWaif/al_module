from small_text.integrations.transformers.classifiers.factories import (
    SetFitClassificationFactory,
    TransformerBasedClassificationFactory,
)
from small_text.integrations.transformers.classifiers.setfit import SetFitModelArguments
from small_text.integrations.transformers.classifiers.classification import (
    TransformerModelArguments,
)
from small_text import PoolBasedActiveLearner, SubsamplingQueryStrategy
from small_text.query_strategies import QueryStrategy
from dataset_class import DatasetLog
from typing import Union

def factory(
    model_name: str,  # The name of the sentence transformer model to be used for classification
    num_classes: int,  # The number of classes in the classification task
    epochs: int,  # The number of epochs to train the classification model
    multi_label: bool,  # Whether the classification task is multi-label or not
    batch: int,  # The batch size to use during training
):
    """
    Creates a classification factory for either SetFit or Transformers-based models, depending on the value of `model_base`.

    Args:
        model_name (str): The name of the sentence transformer model to be used for classification.
        num_classes (int): The number of classes in the classification task.
        epochs (int): The number of epochs to train the classification model.
        multi_label (bool): Whether the classification task is multi-label or not.
        batch (int): The batch size to use during training.
        model_base (str): The type of classification model to use (either "setfit" or "transformers").

    Returns:
        SetFitClassificationFactory: A classification factory for SetFit models.
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
):
    """
    Initializes an active learning process with the given query strategy, dataset, and classification factory.

    Args:
        query (QueryStrategy): The query strategy to be used for active learning.
        dataset (DatasetLog): The dataset to be used for active learning.
        factory (SetFitClassificationFactory): The classification factory to be used for active learning.

    Returns:
        PoolBasedActiveLearner: An active learner object initialized with the given query strategy, dataset, and classification factory.
    """
    query_strategy = SubsamplingQueryStrategy(query)
    active_learner = PoolBasedActiveLearner(
        factory, query_strategy, dataset.textDataset(True)
    )
    return active_learner
