DATA_PATH = "data/"
ENCODE_CONFIG_FILE = 'data/encode_configs.json'

MODELS = [
    'roberta-large-snli_mnli_fever_anli_r1_r2_r3',
    # 'facebook/bart-large-mnli-help',
    # 'facebook/bart-large-mnli',
    # 'bert-base-uncased-snli-help',
    # 'bert-base-uncased-snli',
    # 'roberta-large-mnli-help',
    # 'roberta-large-mnli', 
    # 'infobert', 
]

DATASET_NAMES = [
    'snli',
    'multi_nli',
    'anli'
]

MAX_LENGTH = 256 #128
BATCH_SIZE = 32

TOKENIZERS = {
    'bert-base-uncased-snli': 'textattack/bert-base-uncased-snli',
    'bert-base-uncased-snli-help': 'textattack/bert-base-uncased-snli',
    'roberta-large-mnli': 'roberta-large-mnli',
    'roberta-large-mnli-help': 'roberta-large-mnli',
    'roberta-large-mnli-double_finetuning': 'roberta-large-mnli',
    'facebook/bart-large-mnli': 'facebook/bart-large-mnli',
    'facebook/bart-large-mnli-help': 'facebook/bart-large-mnli',
    'infobert': './models/infobert-checkpoint',
    'roberta-large-snli_mnli_fever_anli_r1_r2_r3': 'ynie/roberta-large-snli_mnli_fever_anli_r1_r2_r3-nli',
}

MODEL_HANDLES = {
    'bert-base-uncased-snli': 'textattack/bert-base-uncased-snli',
    'bert-base-uncased-snli-help': './models/bert-base-uncased-snli-help',
    'roberta-large-mnli': 'roberta-large-mnli',
    'roberta-large-mnli-help': './models/roberta-large-mnli-help',
    'roberta-large-mnli-double_finetuning': './models/roberta-large-mnli-double_finetuning',
    'roberta-large-snli_mnli_fever_anli_r1_r2_r3': 'ynie/roberta-large-snli_mnli_fever_anli_r1_r2_r3-nli',
    'facebook/bart-large-mnli': 'facebook/bart-large-mnli',
    'facebook/bart-large-mnli-help': './models/facebook-bart-large-mnli-help',
    'infobert': './models/infobert-checkpoint',
}

LABEL2ID_2CLASS = {
    "entailment": 0,
    "ENTAILMENT": 0,
    "non-entailment": 1,
    "NON_ENTAILMENT": 1,
    "contradiction": 1,
    "CONTRADICTION": 1,
    "neutral": 1,
    "NEUTRAL": 1,
}

LABEL2ID = {
    'bert-base-uncased-snli': {
        'entailment': 1,
        'neutral': 2, 
        'contradiction': 0
    },
    'infobert': {
        'entailment': 2, 
        'neutral': 1, 
        'contradiction': 0,
    },
    'bert-base-uncased-snli-help': {
        'entailment': 1,
        'neutral': 0, 
        'contradiction': 2
    },

    'roberta-large-mnli': {
        'entailment': 2,
        'neutral': 1, 
        'contradiction': 0
    },
    # no idea actually, check
    'roberta-large-mnli-double-finetuning': {
        'entailment': 1,
        'neutral': 0, 
        'contradiction': 2
    },
    'roberta-large-mnli-help': {
        'entailment': 1,
        'neutral': 0, 
        'contradiction': 2
    },
    # 'roberta-large-mnli-help': {
    #     'entailment': 1,
    #     'neutral': 0, 
    #     'contradiction': 0
    # },
    'facebook/bart-large-mnli-help': {
        'entailment': 2,
        'neutral': 1, 
        'contradiction': 0
    },
    # 'facebook/bart-large-mnli-help': {
    #     'entailment': 2,
    #     'neutral': 1, 
    #     'contradiction': 1
    # },
}


SPLITS = {
    'snli': ['test'],
    'multi_nli': ['validation_matched', 'validation_mismatched'],
    'anli': ['test_r1', 'test_r2', 'test_r3']
    # 'nli_xy': 'test'
}

# DIR_PATH = os.getcwd()

# DATA_PATH = os.path.join(DIR_PATH, "data/")
# ENCODE_CONFIG_FILE = os.path.join(DIR_PATH, 'data/encode_configs.json')
