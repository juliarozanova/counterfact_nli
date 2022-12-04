import random
import os
# from interventions.context_interventions import change_context_same_result
from interventions.insertion_interventions import change_insertions_interventions_same_result
from nli_xy.encoding import build_dataset, parse_encode_config, load_tokenizer
import torch
from experiments.constants import ENCODE_CONFIG_FILE, DATA_PATH

# INTERVENTION_TYPES_TWO_RESULTS = ['0', '1', '1b', '1c', '4']
# INTERVENTION_TYPES_SINGLE_RESULT = ['2', '3', '10', '11']
# INTERVENTION_TYPES = ['change_insertions', 'change_context_same_result', 'change_context_different_result']
INTERVENTION_TYPES = ['0', '1', '2']

def construct_intervention_prompts(intervention_type, encode_config, args):
    '''
    Create model agnostic text representations
    '''
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(encode_config)

    dataset = build_dataset(DATA_PATH, encode_config, tokenizer)
    # counterfactual_contexts_dataset = build_dataset(DATA_PATH, encode_config, tokenizer, counterfactual_contexts=True)

    if intervention_type == '0':
        interventions = change_insertions_interventions_same_result(dataset, change_result=False)

    # if intervention_type == "1":
    #     interventions = change_context_same_result(dataset, counterfactual_contexts_dataset)
    #     pass
    # if intervention_type =="2":
    #     interventions = change_context_different_result(dataset, counterfactual_contexts_dataset)
    # else:
    #     raise Exception('intervention_type not defined {}'.format(intervention_type))
    return interventions