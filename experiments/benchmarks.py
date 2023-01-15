from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, BertTokenizer
from datasets import load_dataset, load_dataset_builder
from experiments.constants import MODELS, MODEL_HANDLES, TOKENIZERS, DATASET_NAMES, SPLITS, LABEL2ID, MAX_LENGTH, BATCH_SIZE
from sklearn.metrics import accuracy_score
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import torch

experiments_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(experiments_dir)


def encode(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

def reinterpret_3_class_as_2_class_labels(int_label, id2label):
    label = id2label(int_label)
    if label in ["entailment", "ENTAILMENT"]:
        return 0
    elif label in ["neutral", "NEUTRAL", "contradiction", "CONTRADICTION"]:
        return 1
    else:
        logger.info(f"Unrecognized label {label}, cannot interpret as two class label!")


def run_eval(model, model_name, dataset_name, split, summary_table):
    logger.info(f'Evaluating on dataset {dataset_name}, {split}:')
    dataset = load_dataset(dataset_name, split=split).filter(lambda x :  x['label']!=-1)

    try:
        label2id = model.config.label2id
        dataset = dataset.align_labels_with_mapping(label2id, 'label')
    except KeyError:
        label2id = LABEL2ID[model_name]
        dataset = dataset.align_labels_with_mapping(label2id, 'label')
    id2label = {v: k for k, v in label2id.items()}
    
    dataset.set_format(type="torch", device='cuda')
    dataset = dataset.map(encode, batched=True)
    dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "premise", "hypothesis",  "labels"], device='cuda')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)


    model.to('cuda')
    model.eval()

    results = {}
    results = pd.DataFrame(results)
    results["y_true"] = dataset['labels'].to('cpu')
    results["y_true"].apply(int)
    results["y_true_2_class"] = results["y_true"].map(lambda int_label: convert_3_class_to_2_class_labels(int_label, id2label))

    with torch.no_grad():
        y_pred = []
        for inputs in tqdm(dataloader):
            batch_outputs = model(inputs['input_ids'], inputs['attention_mask'])
            batch_logits = batch_outputs['logits'].to('cpu')
            batch_predictions = np.argmax(batch_logits, axis=1)
            y_pred += batch_predictions
    
    results["y_pred"] = y_pred
    results["y_pred"] = results["y_pred"].apply(int)
    
    accuracy = accuracy_score(results["y_true"], results["y_pred"])
    logger.info(f'Accuarcy score for {model_name} on {dataset_name}, split \'{split}\': {accuracy} \n')
    prediction_counts = []

    prediction_counts = {
        "y_true_is_1": results.loc[results.y_true==1].y_pred.value_counts(),
        "y_true_is_0": results.loc[results.y_true==0].y_pred.value_counts(),
        "y_true_is_2": results.loc[results.y_true==2].y_pred.value_counts(),
    }

    return accuracy, label2id, prediction_counts

if __name__ == "__main__":

    summary_table = [] 
    for model_name in MODELS:

        logger.info(f'Running benchmark evaluations for the model {model_name}:')
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_HANDLES[model_name], resume_download=True)

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZERS[model_name])

        row = {}
        row['model'] = model_name
        for dataset_name in DATASET_NAMES:
            for split in SPLITS[dataset_name]:
                accuracy, label2id, prediction_counts = run_eval(model, model_name, dataset_name, split, summary_table)
                row[(dataset_name, split)] = accuracy
                row['label2id'] = label2id
                row['prediction_counts'] = prediction_counts

        summary_table.append(row)
    
    summary_table = pd.DataFrame.from_records(summary_table, index='model')
    results_dir = os.path.join(experiments_dir,'results/benchmarks/')
    logger.info(f'\n {summary_table}')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_path = os.path.join(results_dir, 'summary_table.tsv')
    
    with open(results_path, 'w+') as results_file:
        results_file.write(summary_table.to_csv(sep='\t'))