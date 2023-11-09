# al_module
Modulo para utilização do argilla em conjunto com active learning
## Criando venv
``python -m venv venv``
### Ativando a venv
``source venv/bin/activate``

## Instalando dependências
``pip install -r requirements.txt``

## Execução
``python3 main.py``

## Configuração

### O arquivo ``config.json`` é utilizado para configurar todos os aspectos do sistema, as seguinte chave são usadas para configuração:
- ``owner``: contêm as informações necessarias para autenticação no argilla, devem ser definidos a chave de api e workspace de um usuario com permissão de criação de dataset no argilla.
- ``active_learning_config``: é definido o metodo de query utilizado o numero de amostras por batch enviados para o argilla e um numero maximo de iterações.
- ``training_config``: são definidos o numero de epocas por rodada do active learning, o tamanho de lote e se será um treinamento multi-label ou não.
- ``model_config``: é definido o modelo de sentence transformer a ser utilizado no treinamento.
- ``dataset_config``: é definido o nome do dataset no argilla e o caminho do arquivo ``.csv`` com os dados a serem utilizados.
  
  **OBS:** o nome do dataset argilla deve conter apenas letras minusculas e sem caracteres especiais.
- ``inputs``: estrutura o dados a serem mostrados no argilla, a chave de cada campo será o nome utilizado no argilla e o valor é a coluna correspondente no ``.csv`` definido.

  **Exemplo:** ``"inputs" : {
        "OBJETO":"text",
        "ID-LICITACAO":"_ID-LICITACAO",
        "ID-ARQUIVO":"_ID-ARQUIVO"
    }``, cada registro no argilla terá esses três campos ``OBJETO,ID-LICITACAO,ID-ARQUIVO`` e as colunas correspondentes no dataframe são ``text,_ID-LICITACAO,_ID-ARQUIVO``
- ``training_field``: define qual dos campos informado em input será utilizado para treinamento
- ``log_path``: define o caminho para os arquivos de log
- ``pretraining``: define se será feito ou não pre-treinamento, a busca por dados de pre-treinamento é feito no proprio dataframe, caso haja registro com a coluna ``_isSend`` igual a ``True`` esses dados serão utilizados como pre-treinamento sem a necessidade de serem passados para o argilla, essa configuração foi implementada no caso de haver registros já rotulados sem a necessidade de manda-los para o argilla e resgata-los de volta.
- ``training_labels``: define quais label serão usada para treinamento, as labels todas as labels são resgatadas automaticamente das colunas presentes no dataframe, somente não serão labels a coluna ``text`` e todas as outras coluna iniciadas com o prefixo ``_``, esse conjunto de labels é utilizado para gerar o dataset no argilla. No entanto não necessariamente todas elas precisam ser utilizadas para treino, aquelas que forem utilizadas precisam ser passadas nesse parametro.

  **OBS:** Caso queira adicionar qualquer coluna do dataframe basta adicionar o prefixo ``_`` para a coluna ser ignorada.

- ``metricas_pth``: define a pasta onde será salvo os arquivo de metricas, são três arquivos gerados ``metricas_front.csv`` que avaliam o modelo para o proximo lote rotulado que ainda não foi treinado pelo active_learning a cada rodada, ``metricas_back.csv`` avalia para todo o conjunto de dados já treinado, ``metricas_teste.csv`` avalia o modelo para um conjunto de teste predefinido.
  
  **OBS:** Atualmente as metricas calculadas são: *acuracia,fi-score,hamming-loss,precisão,recall e confiança de pesquisa*

- ``model_pth``: define a pasta onde o modelo será salvo, a cada rodada de active learning uma nova versão do modelo é salva.
- ``teste_pth``: define o caminho do arquivo de teste, o arquivo de teste deve ser um arquivo de bytes ``pickle`` onde os dados de teste estão previamente rotulados e no formato *TextDataset*.

  **Exemplo:**
  
  ```
  from test_construct import createTextDataset
  createTextDataset('test_ml.csv',["label_1","label_2"],True)
  ```
  ```
  (function) def createTextDataset(
    path_test: str,
    training_labels: list,
    multi_label: bool
  ) -> None
  Creates a text dataset from a CSV file and saves it as a pickle file.
  
  Args:
  
  path_test (str): The path to the CSV file.
  training_labels (list): A list of labels for the training data.
  multi_label (bool): A boolean indicating whether the dataset has multiple labels.
  Returns:
  
  None
  ```
- ``url``: link para uma instancia do argilla funcional a ser utilizada.
- ``workspace_user``: workspace utilizado para enviar os dados de casa conjunto, note que o workspace já deve existir e usarios pertencentes a ele poderão rotular os dados.
- ``data_storage``: define a pasta onde os dados rotulados serão salvos, ``registro.csv`` mantêm apenas os registro rotulados onde _isSend será true, ``historico.csv`` mantêm os registros rotulados e não rotulados.
- ``earlyStopping``*(opcional)*: define o comportamento de earlyStopping, os parametros possivels são *target_metric,patience,threshold* , caso a metrica assistida em *target_metric* não melhore depois de *patience* rodadas, o treinamento é parado. No caso de *threshold* se a metrica ultrapassar esse valor o treinamento é iterrompido, ambos *patience* e *threshold* são opcionais mas caos nenhum seja informado nunca havera condição de parada de earlyStopping.
- ``active_learning`` : define se de fato será usado active learning, caso seja false todos os dados serão enviados juntos para o argilla e não acontecerá treinamento.
  
  **OBS** Foi considerado o comportamento da metrica ``hamming loss`` que funciona de forma inversa a das outras.

## Dataframe exemplo

text|_link|_dt_criacao|label_1|label_2
----|-----|-----------|-------|-------
"Algo deve está aqui" | (http://example.com) |2022-05-10| 0 | 0
