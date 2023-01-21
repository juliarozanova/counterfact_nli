from experiments.constants import LABEL2ID, LABEL2ID_2CLASS, MAX_LENGTH
from loguru import logger

def encode(example, tokenizer):
    return tokenizer(example["premise"], example["hypothesis"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

def map_3_class_to_2_class_preds(int_label, id2label):
    string_label = id2label[int_label]

    try:
        return LABEL2ID_2CLASS[string_label]
    except KeyError:
        logger.info(f"Unrecognized label {string_label}, cannot interpret as two class label!")

def get_label_maps(model, model_name):
    ''' Get two label dictionaries which allow for interpretation of the model's inputs. 

    Parameters
    ----------
    model : transformers.AutoModelForSequenceClassification
        loaded transformers sequence classification model
    model_name : str
        shorthand model name descriptor, e.g. 'roberta-large-mnli'
    LABEL2ID : dict
        global backup dict storing label maps for models that may not have an up-to-date/properly formatted label map

    Returns
    -------
    label2id: dict
        maps 3 class entailment labels (neutral, entailment, contradiction) to integers, as per the model's training regime
    id2label: dict
        maps integer labels to 3 class entailment string labels (neutral, entailment, contradiction), as per the model's training regime
    '''
    try: 
        label2id = model.config.label2id
        assert ("entailment" in label2id.keys()) or ("ENTAILMENT" in label2id.keys())
    except AssertionError:
        label2id = LABEL2ID[model_name]
    
    # standardise case
    label2id = {k.lower():v for k, v in label2id.items()}

    # inverse dict
    id2label = {v: k for k, v in label2id.items()}

    logger.info(f"Model name: {model_name}, Model label scheme: {label2id}")
    return label2id, id2label