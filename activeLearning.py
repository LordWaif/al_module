from dataset_class import DatasetLog
from small_text import PoolBasedActiveLearner,TextDataset
import argilla,json,os
from pathlib import Path
from argilla.listeners import listener
from utils import transformY,createTD,hot_encode,_hot_encode,predict
import numpy as np
from metrics import calcule_metrics
import pandas as pd
import uuid
from argilla_functions import createRecords

metricas_front = {'acc':[],'f1':[],'hs':[],'precision':[],'recall':[],'batch':[],'confidence':[],'confidence_query':[]}
metricas_back = {'acc':[],'f1':[],'hs':[],'precision':[],'recall':[],'batch':[],'confidence':[],'confidence_query':[]}
metricas_teste = {'acc':[],'f1':[],'hs':[],'precision':[],'recall':[],'batch':[],'confidence':[],'confidence_query':[]}

def execute(dataset: DatasetLog, active_learning_base: PoolBasedActiveLearner, rg: argilla,data_json:dict,logger):
    logger.info(f"Starting Active Learning for {dataset.name}")
    @listener(
        dataset=dataset.name,
        query="status:Validated AND metadata.batch_id:{batch_id}",
        condition=lambda search: search.total == dataset.num_samples,
        execution_interval_in_seconds=3,
        batch_id=0,
        isInitial=True,
        qtd=dataset.num_samples,
    )
    def active_learning_loop(records, ctx):
        global metricas_front,metricas_back,metricas_teste
        logger.info(f"Updating with batch_id {ctx.query_params['batch_id']}")
        dados = rg.load(
                name=dataset.name, query=f"status:Validated AND metadata.batch_id:<={ctx.query_params['batch_id']}")
        if ctx.query_params["isInitial"] and not(data_json['pretraining']):
            logger.info("Pretraining is disabled")
            y = transformY(dados, dataset.LABEL2INT,
                            len(data_json['training_labels']), dataset.multi_label,data_json['training_labels'])
            ctx.query_params["isInitial"] = False
            indices = np.array([_.id for _ in dados])
            active_learning_base.initialize_data(indices, y)
        else:
            logger.info(f"Avaliando modelo para o ultimo lote")
            y_true,y_pred,proba_acc = predict(al=active_learning_base,data_json=data_json,data=records,dataset=dataset)
            metricas_front = calcule_metrics(metricas_front,y_true,y_pred,proba_acc,ctx.query_params['batch_id'])
            metricas_front['confidence_query'].append(np.mean(active_learning_base.query_strategy.base_query_strategy.scores_))
            pd.DataFrame(metricas_front).to_csv(os.path.join(data_json['metricas_pth'],'metricas_front.csv'),index=False)
            if isinstance(active_learning_base.indices_queried,type(None)):
                active_learning_base.indices_queried = np.array([_.id for _ in records])
            y = transformY(records, dataset.LABEL2INT,
                            len(data_json['training_labels']), dataset.multi_label,data_json['training_labels'])
            logger.info("Training")
            active_learning_base.update(y)
            logger.info(f"Finished training")
            # --------------------
            # Calculo de metricas back
            logger.info(f"Calculando metricas para todo o conjunto já treinado")
            y_true,y_pred,proba_acc = predict(al=active_learning_base,data_json=data_json,data=dados,dataset=dataset)
            metricas_back= calcule_metrics(metricas_back,y_true,y_pred,proba_acc,ctx.query_params['batch_id'])
            metricas_back['confidence_query'].append(-1)
            pd.DataFrame(metricas_back).to_csv(os.path.join(data_json['metricas_pth'],'metricas_back.csv'),index=False)
            # --------------------
        logger.info(f"Salvando modelo")
        active_learning_base.save(os.path.join(data_json['model_pth'],f'modelo_{ctx.query_params["batch_id"]}_{uuid.uuid1().__str__()}_.pkl'))
        # TESTE_COM_ROTULADOS
        import pickle
        with open(data_json['teste_pth'],'rb') as _file_teste:
            teste = pickle.load(_file_teste)
        csr_teste, proba_teste = active_learning_base.classifier.predict(
            teste, return_proba = True
        )

        y_pred_teste = hot_encode(csr_teste,len(data_json['training_labels']),dataset.multi_label)
        y_true_teste = hot_encode(teste.y,len(data_json['training_labels']),dataset.multi_label)
        metricas_teste = calcule_metrics(metricas_teste,y_true_teste,y_pred_teste,proba_teste,ctx.query_params['batch_id'])
        metricas_teste['confidence_query'].append(-1)
        pd.DataFrame(metricas_teste).to_csv(os.path.join(data_json['metricas_pth'],'metricas_teste.csv'),index=False)
        # ----------------------
        logger.info(f"Saving {len(records)} records")
        YtoSave = hot_encode(transformY(records, dataset.LABEL2INT,
            dataset.num_classes, dataset.multi_label,dataset.LABELS), dataset.num_classes, dataset.multi_label)
        dataset._dataset.loc[[_records.id for _records in records],dataset.LABELS] = YtoSave
        dataset._dataset.to_csv(os.path.join(data_json['data_storage'],'historico.csv'),index=False)
        if ctx.query_params["batch_id"] == 0:
            dataset._dataset.loc[[_records.id for _records in records]].to_csv(os.path.join(data_json['data_storage'],'registros.csv'),index=False)
        else:
            dataset._dataset.loc[[_records.id for _records in records]].to_csv(os.path.join(data_json['data_storage'],'registros.csv'),mode='a',header=False,index=False)

        # QUERY
        logger.info(f"Querying for {dataset.num_samples} samples")
        queried_indices = active_learning_base.query(
            num_samples=dataset.num_samples
        )
        dataset._dataset.loc[queried_indices, "_isSend"] = True

        # Create new package to send
        new_batch = ctx.query_params["batch_id"] + 1
        new_records = createRecords(
                dataset, new_batch, queried_indices, rg,data_json['inputs']
        )
        records_textDataset: TextDataset = createTD(data_json['training_labels'],
            new_records, dataset, data_json['training_field'])
        _, proba = active_learning_base.classifier.predict(
            records_textDataset, return_proba=True
        )
        probabilities = []
        for i in proba:
            l = []
            for lb,j in zip(data_json['training_labels'],i):
                l.append([lb,j])
            probabilities.append(l)
        restLabels = [i for i in dataset.LABELS+['__control__'] if i not in data_json['training_labels']]
        for i in range(len(probabilities)):
            probabilities[i]+=[[j,0.0] for j in restLabels]
            probabilities[i] = [tuple(i) for i in probabilities[i]]
        for i in range(len(new_records)):
            new_records[i].prediction = probabilities.pop(0)
        rg.log(records=new_records, name=dataset.name,workspace='victor_silva')
        ctx.query_params["batch_id"] = new_batch
    return active_learning_loop