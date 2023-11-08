import json,os
from dataset_class import DatasetLog

data_json = json.load(open("config.json"))
os.makedirs(data_json['log_path'], exist_ok=True)
os.makedirs(data_json['metricas_pth'], exist_ok=True)
os.makedirs(data_json['model_pth'], exist_ok=True)
os.makedirs(data_json['data_storage'], exist_ok=True)

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
    if data_json['active_learning']:
        active_learning_base = \
            initialize_activeLearning(
                query,
                dataset,
                factory(
                    dataset.model_name,
                    2 if len(data_json["training_labels"]) == 1 else len(data_json["training_labels"]),
                    dataset.epochs,
                    dataset.multi_label,
                    dataset.batch,
                ),
            )
    def pre_training():
        pre_data = dataset.dataset[dataset.dataset['_isSend'] == True]
        #pre_data = pre_data.sample(frac=0.05)
        indices = np.array(pre_data.index.to_list())
        if indices.shape[0] == 0:
            logger.info("No data to pre training")
            raise Exception("No data to pre training")
        y = pre_data[data_json["training_labels"]].values
        if data_json['training_config']['multi_label']:
            from small_text import list_to_csr
            from utils import label_encode
            y = label_encode(y)
            y = list_to_csr(y,shape=(y.shape[0],len(data_json["training_labels"])))

        active_learning_base.initialize_data(indices,y)
        active_learning_base.save(os.path.join(data_json['model_pth'],f'modelo_pre_training_{uuid.uuid1().__str__()}_.pkl'))
    if data_json['pretraining'] and data_json['active_learning']:
        pre_training()
        import pickle,pandas as pd
        from metrics import calcule_metrics
        with open(data_json['teste_pth'],'rb') as _file_teste:
            teste = pickle.load(_file_teste)
        csr_teste, proba_teste = active_learning_base.classifier.predict(
                teste, return_proba = True
            )
        y_pred = teste.y
        metricas_teste = {'acc':[],'f1':[],'hs':[],'precision':[],'recall':[],'batch':[],'confidence':[],'confidence_query':[]}
        metricas_teste = calcule_metrics(metricas_teste,y_pred,csr_teste,proba_teste,-1)
        metricas_teste['confidence_query'].append(-1)
        pd.DataFrame(metricas_teste).to_csv(os.path.join(data_json['metricas_pth'],'metricas_teste_pre_training.csv'),index=False)
    import argilla as rg
    rg.init(
        api_url= data_json["url"],
        api_key= data_json["owner"]["api_key"],
        workspace= data_json["owner"]["workspace"]
    )
    from argilla_functions import initialize_log
    from activeLearning import execute
    initialize_log(dataset, rg, 'victor_silva', data_json["inputs"],data_json['active_learning'])
    rg.active_client().set_workspace(data_json["workspace_user"])
    if data_json['active_learning']:
        function = execute(dataset, active_learning_base,
                            rg, data_json,logger)
    else:
        function = execute(dataset, None,
                            rg, data_json,logger)
    function.start()
    function.__current_thread__.join()
    rg.load(name=dataset.name).to_pandas().to_csv(os.path.join(data_json['data_storage'],'backup_argilla.csv'),index=False)