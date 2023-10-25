import pandas as pd
import random as rd
from typing import Union
from sklearn.utils import resample
from numpy import array, arange
from small_text import TextDataset
from small_text.utils.labels import list_to_csr, csr_matrix

class DatasetLog:
    def __init__(self):
        """
        Initializes a new instance of the DatasetLog class.

        Args:
            num_samples (int, optional): The number of samples to be randomly selected for active learning. Defaults to NUM_SAMPLES.
            multi_label (bool, optional): Whether the dataset has multiple labels or not. Defaults to True.
        """
        # active learning
        self._max_interactions = None
        self._query = None
        # dataset
        self._name = None
        self._pth = None
        # training
        self._tds = None
        
    def set_model_config(self,**kwargs):
        self._model_name = kwargs.get("model_name")

    def set_training_config(self,**kwargs):
        self._epochs = kwargs.get("epochs")
        self._batch = kwargs.get("batch")
        self._multi_label = kwargs.get("multi_label")

    def set_active_learning_config(self,**kwargs):
        self._num_samples = kwargs.get("num_samples")
        self._max_interactions = kwargs.get("max_interactions")
        self._query = kwargs.get("query")

    def set_dataset_config(self,**kwargs):
        '''
        Sets the configuration for the dataset.

        Args:
            **kwargs: keyword arguments that can be passed to set the dataset configuration. The following arguments are available:
                pth (str): path to the dataset file.
                name (str): name of the dataset.
                validation (str): path to the validation dataset file.
        '''
        self._pth = kwargs.get("pth")
        self._name = kwargs.get("name")
        if 'validation' in kwargs:
            self._pth_valid = kwargs.get("validation")
            self._validation = pd.read_csv(self._pth_valid)
        self._dataset = pd.read_csv(self._pth)
        try:
            self._dataset.insert(self._dataset.shape[1], "_isSend", False)
        except ValueError:
            pass
        if "text" not in list(self._dataset.columns):
            raise AttributeError(f'Column "text" not exist')

    def set_label_function(self,labels: list = None):
        if not labels:
            self.LABELS = [_ for _ in list(self._dataset.columns) if not _.startswith('_') and not _ == 'text']
        else:
            self.LABELS = labels
        
    def set_argilla_config(self,**kwargs):
        self._user = kwargs.get("user")
        self._allowed_users = kwargs.get("allowed_users")

    def __str__(self):
        return (
            f"DatasetLog<{self._name},"
            f"len={len(self._dataset)},"
            f"classes={self.num_classes},"
            f"multi_label={self._multi_label},"
            f"labels=({self.LABELS})>"
        )
    
    def random_sample(self) -> list:
        indices =  rd.sample(self._dataset[self._dataset['_isSend']==False].index.to_list(),self._num_samples)
        self._dataset.loc[indices, "_isSend"] = True
        return indices
    
    @property
    def query(self) -> str:
        return self._query
    
    @query.setter
    def query(self, new_query: str) -> None:
        self._query = new_query
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, new_name: str) -> None:
        self._name = new_name

    def validation_split(self, _validation_split: float) -> None:
        '''
        Divide o dataset em treino e validação de acordo com a proporção definida em _validation_split.

        Args:
            _validation_split (float): proporção do dataset que será usada para validação. Deve ser um float no intervalo [0,1].

        Raises:
            ValueError: se _validation_split for menor que 0 ou maior ou igual a 1.

        '''
        if not self._validation:
            indices = self._dataset.shape[0]
            if _validation_split < 0 or _validation_split >= 1:
                raise ValueError(
                    f"Invalid value:{_validation_split}, validation split must be a float in interval [0,1]"
                )
            _split = round(indices * _validation_split)
            train = self._dataset[_split:]
            validation = self._dataset[:_split]

            self._dataset = train
            self._validation = validation

    @property
    def multi_label(self) -> bool:
        '''
            Retorna se o dataset é multi-label ou não.
        '''
        return self._multi_label
    
    @multi_label.setter
    def multi_label(self, new_multi_label: bool) -> None:
        '''
            Define se o dataset é multi-label ou não.
        '''
        self._multi_label = new_multi_label
            
    @property
    def validation(self) -> pd.DataFrame:
        '''
            Retorna o dataset de validação.
        '''
        if hasattr(self,'_validation'):
            raise ValueError("Validation split or dataset is not defined.")
        return self._validation

    @property
    def max_interactions(self) -> int:
        return self._max_interactions

    @max_interactions.setter
    def max_interactions(self, new_max_intections) -> None:
        self._max_interactions = new_max_intections

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, new_model_name: str) -> None:
        self._model_name = new_model_name

    @property
    def epochs(self) -> int:
        return self._epochs

    @epochs.setter
    def epochs(self, new_epoch: int) -> None:
        self._epochs = new_epoch

    @property
    def batch(self) -> int:
        return self._batch

    @batch.setter
    def batch(self, new_batch: int) -> None:
        self._batch = new_batch

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @num_samples.setter
    def num_samples(self, new_num_samples: int) -> None:
        self._num_samples = new_num_samples

    @property
    def user(self) -> str:
        return self._user

    @user.setter
    def users(self, new_users) -> None:
        self._user = new_users

    @property
    def id(self) -> str:
        """
        Retorna um concatenção do usuário e nome do argilla que é utilizada como id.

        Exemplos:
        ---------
        DatasetLog.user = 'admin'
        DatasetLog.name = 'database'
        DatasetLog.id -> 'admin.database'

        Parâmetros:
        -----------
        Nenhum.

        Retorno:
        --------
        f'{self.user}.{self._name}' : str
        """
        return f"{self.user}.{self._name}"

    @property
    def shape(self) -> tuple:
        return self._dataset.shape

    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset

    def sample(self, frac=1, inplace=False) -> Union[pd.DataFrame, None]:
        """
        Retorna uma amostra aleatória do dataset. Caso seja usada com frac = 1, apenas retorna
        dataset com a ordem dos registros aleatória.

        Parâmetros:
        -----------
        inplace : bool, default = False
            Se False, retorna uma cópia. Senão, faz a operação de forma implícita e retorna None.

        frac : float, default = 1
            Determina a porcentagem do dataset que será usada na amostra, 1 corresponde a 100%.

        Retorno:
        --------
        __dataset : pd.DataFrame ou None
        """
        self.__dataset = self._dataset.sample(frac=frac).reset_index(drop=True)
        if not inplace:
            return self.__dataset
        else:
            self.bcp = self.dataset.copy()
            self._dataset = self.__dataset
            self.__dataset = self.bcp
            return

    def sample_balanced(self, inplace=False) -> Union[pd.DataFrame, None]:
        """
            Retorna uma amostra balanceada do dataset, ela têm como objetivo equalizar a quantidade

            de elementos de cada classe. Ou seja tenta manter o numero de classes proximo.

        Parametros:
        -----------
            inplace : bool, default = False
            Se false, retorna uma copia. Senão, faz a operação de forma implicita e retorna None.

        Retorno:
        -----------
            _dataset.reset_index(drop=True) : pd.DataFrame
        """
        counts = self._dataset[self._LABELS].sum()
        min_count = counts.min()
        balanced_data = []
        for label_column in self._LABELS:
            data_with_label = self._dataset[self._dataset[label_column] == 1]
            balanced_label_data = resample(
                data_with_label, n_samples=min_count, random_state=42
            )
            balanced_data.append(balanced_label_data)
        balanced_data = pd.concat(balanced_data)
        balanced_data = balanced_data.sample(frac=1, random_state=42)
        balanced_data.drop_duplicates(inplace=True)
        self.__dataset = balanced_data
        if not inplace:
            return self.__dataset.reset_index(drop=True)
        else:
            self.bcp = self.dataset.copy()
            self._dataset = self.__dataset
            self._dataset.reset_index(drop=True, inplace=True)
            self.__dataset = self.bcp

    @property
    def unsample(self) -> None:
        self._dataset = self.__dataset

    def _searchLabels(self, row: pd.Series) -> array:
        rotulos = []
        for c in self.LABELS:
            if row[c] == 1 and c != "annotation":
                rotulos.append(c)
        if self._multi_label:
            return rotulos
        else:
            return rotulos[0]

    def _labeling(self, inplace: bool = True, x:pd.DataFrame = None) -> Union[pd.Series, None]:
        if x is None:
            x = self.__dataset
        if inplace:
            self._dataset["annotation"] = x.apply(
                self._searchLabels, axis=1
            )
        else:
            return x.apply(self._searchLabels, axis=1)

    def annotation(self, inplace: bool = True, x:pd.DataFrame = None) -> pd.Series:
        return self._labeling(inplace,x)

    @property
    def LABELS(self) -> list:
        return self._LABELS

    @LABELS.setter
    def LABELS(self, labels: list):
        self._LABELS = labels
        for label in self._LABELS:
            try:
                self._dataset.insert(self._dataset.shape[1], label, 0)
            except ValueError:
                ...

    @property
    def num_classes(self) -> int:
        return len(self.LABELS)

    def one_hot_encoding(self) -> array:
        """
            Recupera as labels de todos os registros em _dataset, como _dataset é um pd.DataFrame

            é feito um filtro para as colunas em LABELS e depois executado pd.Dataframe.values para

            retornar o valor em np.array.

        Parametros:
        ----------
            -
        Retorno:
        ----------
            self._dataset[self.LABELS].values : np.array
        """
        return self._dataset[self.LABELS].values

    def y(self, isTrain: bool = False, x: pd.DataFrame = None) -> Union[csr_matrix, array]:
        """
            Realiza o tratamento das possiveis labels dos dados, dependendo do valor de isTrain.

            As labels podem ser vazias ou conter os dados, caso seja multi-label o retono é uma csr_matrix,

            para outros casos é um array numpy.

        Parametros:
        ----------
            isTrain : bool, default = False
            Se falso, retorna y sem anotação. Senão, calcula as anotações para cada registro.

        Retorno:
        ----------
            y : scipy.csr_matrix ou np.array
            retorna as verdadeiras labels com anotação se isTrain for true. Senão retorna sem anotação.
        """
        if x is None:
            x = self._dataset
        if isTrain:
            indexers = []
            for i in self.annotation(False,x):
                indexers.append(self.LABEL2INT(i))
            _y = indexers
        else:
            _y = [[] for i in range(x.shape[0])]
        if self._multi_label:
            return list_to_csr(_y, shape=(len(_y), self.num_classes))
        else:
            return array(_y)

    def textDataset(
        self, isTrain: bool = False
    ) -> TextDataset:
        if self._tds == None:
            self._dataset.reset_index(drop=True, inplace=True)
            self._tds = TextDataset(
                self._dataset["text"].values,
                self.y(isTrain=isTrain,x=self._dataset),
                target_labels=arange(self.num_classes),
            )
        return self._tds

if __name__ == "__main__":
    ...