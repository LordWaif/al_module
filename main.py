import json,os
from dataset_class import DatasetLog

data_json = json.load(open("config.json"))
os.makedirs(data_json['log_path'], exist_ok=True)
os.makedirs(data_json['metricas_pth'], exist_ok=True)
os.makedirs(data_json['model_pth'], exist_ok=True)

# LOGGING
import logging
from pytz import timezone
from datetime import datetime
tz = timezone('America/Fortaleza')
logger = logging.getLogger('al_logger')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(data_json['log_path'],'logActiveLearning.log'))
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter.converter = lambda *args: datetime.now(tz).timetuple()
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

if __name__ == '__main__':
    from model_conf import factory, initialize_activeLearning
    from configuration import queries
    import numpy as np,os,uuid
    data_json = json.load(open("config.json"))
    dataset = DatasetLog()
    dataset.set_training_config(**data_json['training_config'])
    dataset.set_model_config(**data_json['model_config'])
    dataset.set_dataset_config(**data_json['dataset_config'])
    dataset.set_label_function()
    dataset.set_active_learning_config(**data_json['active_learning_config'])
    query = queries[dataset.query]()
    active_learning_base = \
        initialize_activeLearning(
            query,
            dataset,
            factory(
                dataset.model_name,
                2 if len(data_json["training_labels"]) == 1 else len("training_labels"),
                dataset.epochs,
                dataset.multi_label,
                dataset.batch,
            ),
        )
    def pre_training():
        pre_data = dataset.dataset[dataset.dataset['_isSend'] == True]
        #pre_data = pre_data.sample(frac=0.05)
        indices = np.array(pre_data.index.to_list())
        active_learning_base.initialize_data(indices,pre_data[data_json["training_labels"]].values.reshape(-1))
        active_learning_base.save(os.path.join(data_json['model_pth'],f'modelo_pre_training_{uuid.uuid1().__str__()}_.pkl'))
    if data_json['pretraining']:
        pre_training()
        import pickle,pandas as pd
        from metrics import calcule_metrics
        with open(data_json['teste_pth'],'rb') as _file_teste:
            teste = pickle.load(_file_teste)
        csr_teste, proba_teste = active_learning_base.classifier.predict(
                teste, return_proba = True
            )
        metricas_teste = {'acc':[],'f1':[],'hs':[],'precision':[],'recall':[],'batch':[],'confidence':[],'confidence_query':[]}
        metricas_teste = calcule_metrics(metricas_teste,teste.y,csr_teste,proba_teste,-1)
        metricas_teste['confidence_query'].append(-1)
        pd.DataFrame(metricas_teste).to_csv(os.path.join(data_json['metricas_pth'],'metricas_teste_pre_training.csv'),index=False)
    import argilla as rg
    rg.init(
        api_url= data_json["url"],
        api_key= data_json["owner"]["key"],
        workspace= data_json["owner"]["workspace"],
    )
    from argilla_functions import initialize_log
    from activeLearning import execute
    initialize_log(dataset, rg, 'victor_silva', data_json["inputs"])
    rg.active_client().set_workspace(data_json["workspace_user"])
    function = execute(dataset, active_learning_base,
                        rg, data_json["inputs"])
    function.start()
    function.__current_thread__.join()