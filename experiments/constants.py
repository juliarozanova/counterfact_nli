
DATA_PATH = "data/"
ENCODE_CONFIG_FILE = 'data/full_encode_configs.json'

MODELS = [
    'bert-base-uncased-snli',
    'bert-base-uncased-snli-help',
    'roberta-large-mnli', 
    'roberta-large-mnli-help',
    'facebook/bart-large-mnli',
    'facebook/bart-large-mnli-help',
]

THREE_CLASS_MODELS = [
    'bert-base-uncased-snli',
    'roberta-large-mnli', 
    'facebook/bart-large-mnli',
]

TWO_CLASS_MODELS = [
    'bert-base-uncased-snli-help',
    'roberta-large-mnli-help',
    'facebook/bart-large-mnli-help',
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
    'facebook/bart-large-mnli-help': 'facebook/bart-large-mnli'
}

MODEL_HANDLES = {
    'bert-base-uncased-snli': 'textattack/bert-base-uncased-snli',
    'bert-base-uncased-snli-help': './models/bert-base-uncased-snli-help',
    'roberta-large-mnli': 'roberta-large-mnli',
    'roberta-large-mnli-help': './models/roberta-large-mnli-help',
    'roberta-large-mnli-double_finetuning': './models/roberta-large-mnli-double_finetuning',
    'facebook/bart-large-mnli': 'facebook/bart-large-mnli',
    'facebook/bart-large-mnli-help': './models/facebook-bart-large-mnli-help'
}

LABEL2ID = {
    'bert-base-uncased-snli': {
        'entailment': 1,
        'neutral': 2, 
        'contradiction': 0
    },

    'bert-base-uncased-snli-help': {
        'entailment': 1,
        'neutral': 2, 
        'contradiction': 2
    },
    'roberta-large-mnli': {
        'entailment': 2,
        'neutral': 1, 
        'contradiction': 0
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