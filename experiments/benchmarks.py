from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, BertTokenizer
from datasets import load_dataset, load_dataset_builder
from experiments.constants import MODELS, MODEL_HANDLES, TOKENIZERS, DATASET_NAMES, SPLITS, LABEL2ID, LABEL2ID_2CLASS, BATCH_SIZE
from experiments.utils import get_label_maps, map_3_class_to_2_class_preds, encode
from sklearn.metrics import accuracy_score
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import torch

experiments_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(experiments_dir)


def run_eval(model, model_name, dataset_name, split, summary_table):
    logger.info(f'Evaluating on dataset {dataset_name}, {split}:')
    dataset = load_dataset(dataset_name, split=split).filter(lambda x :  x['label']!=-1)

    label2id, id2label = get_label_maps(model, model_name)
    
    # assign ints to dataset gold_labels according to the model's labeling scheme
    dataset = dataset.align_labels_with_mapping(label2id, 'label')
    dataset.set_format(type="torch", device='cuda')
    dataset = dataset.map(lambda example: encode(example, tokenizer), batched=True)
    dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "premise", "hypothesis",  "labels"], device='cuda')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)


    model.to('cuda')
    model.eval()

    results = {}
    results = pd.DataFrame(results)
    results["y_true_3_class"] = dataset['labels'].to('cpu')
    results["y_true_3_class"].apply(int)
    results["y_true_2_class"] = results["y_true_3_class"].map(lambda int_label: LABEL2ID_2CLASS[id2label[int_label]])

    with torch.no_grad():
        y_pred = []
        for inputs in tqdm(dataloader):
            batch_outputs = model(inputs['input_ids'], inputs['attention_mask'])
            batch_logits = batch_outputs['logits'].to('cpu')
            batch_predictions = np.argmax(batch_logits, axis=1)
            y_pred += batch_predictions
    
    results["y_pred_3_class"] = y_pred
    results["y_pred_3_class"] = results["y_pred_3_class"].apply(int)
    results["y_pred_2_class"] = results["y_pred_3_class"].apply(lambda x: map_3_class_to_2_class_preds(x, id2label=id2label))
    
    accuracy_2_class = accuracy_score(results["y_true_2_class"], results["y_pred_2_class"])
    accuracy_3_class = accuracy_score(results["y_true_3_class"], results["y_pred_3_class"])

    logger.info(f'Accuarcy score for {model_name} on {dataset_name}, split \'{split}\': {accuracy_2_class} \n')

    # prediction_counts = {
    #     "y_true_is_1": results.loc[results.y_true==1].y_pred.value_counts(),
    #     "y_true_is_0": results.loc[results.y_true==0].y_pred.value_counts(),
    #     "y_true_is_2": results.loc[results.y_true==2].y_pred.value_counts(),
    # }

    return accuracy_2_class, accuracy_3_class 

if __name__ == "__main__":

    results_dir = os.path.join(experiments_dir,'results/benchmarks/')
    results_path = os.path.join(results_dir, 'summary_table.tsv')

    try: 
        summary_table = pd.read_csv(results_path, sep='\t', header=0).to_dict('records')
    except FileNotFoundError:
        summary_table = [] 

    for model_name in MODELS:
        logger.info(f'Running benchmark evaluations for the model {model_name}:')
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_HANDLES[model_name], resume_download=True)

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZERS[model_name])

        row = {}
        row['model'] = model_name
        for dataset_name in DATASET_NAMES:
            for split in SPLITS[dataset_name]:
                accuracy_2_class, accuracy_3_class = run_eval(model, model_name, dataset_name, split, summary_table)
                row[("accuracy_2_class", (dataset_name,split))] = accuracy_2_class
                row[("accuracy_3_class", (dataset_name,split))] = accuracy_3_class

        summary_table.append(row)
    
    summary_table = pd.DataFrame.from_records(summary_table, index='model')
    logger.info(f'\n {summary_table}')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    
    with open(results_path, 'w+') as results_file:
        results_file.write(summary_table.to_csv(sep='\t'))