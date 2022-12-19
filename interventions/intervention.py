import random
import os
from interventions.insertion_interventions import change_insertions_interventions, change_context_interventions
from nli_xy.encoding import build_dataset, parse_encode_config, load_tokenizer
from loguru import logger
import torch
from experiments.constants import DATA_PATH

# INTERVENTION_TYPES_TWO_RESULTS = ['0', '1', '1b', '1c', '4']
# INTERVENTION_TYPES_SINGLE_RESULT = ['2', '3', '10', '11']
# INTERVENTION_TYPES = ['change_insertions', 'change_context_same_result', 'change_context_different_result']

def construct_intervention_prompts(intervention_type, encode_config, args):
    '''
    Create model agnostic text representations
    '''
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(encode_config)

    dataset = build_dataset(DATA_PATH, encode_config, tokenizer)

    if intervention_type == '0':
        logger.info('Performing intervention: change insertion, same result')
        interventions = change_insertions_interventions(dataset, tokenizer, change_result=False)
    if intervention_type == '1':
        logger.info('Performing intervention: change insertion, change result')
        interventions = change_insertions_interventions(dataset, tokenizer, change_result=True)
    if intervention_type == '2':
        logger.info('Performing intervention: change context, same result')
        interventions = change_context_interventions(dataset, tokenizer, change_result=False)
    if intervention_type == '3':
        logger.info('Performing intervention: change context, change result')
        interventions = change_context_interventions(dataset, tokenizer, change_result=True)
    # else:
    #     raise Exception(f'intervention_type not defined {intervention_type}')

    return interventions