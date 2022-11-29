from data_utils import load_contexts
import torch
from interventions.context_interventions import *
from interventions.insertion_interventions import *

# INTERVENTION_TYPES_TWO_RESULTS = ['0', '1', '1b', '1c', '4']
# INTERVENTION_TYPES_SINGLE_RESULT = ['2', '3', '10', '11']
# INTERVENTION_TYPES = ['change_insertions', 'change_context_same_result', 'change_context_different_result']
INTERVENTION_TYPES = ['0', '1', '2']


def construct_intervention_prompts(tokenizer, intervention_type, args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    templates = construct_numerical_templates(path_to_data=args.path_to_num_data)
    contexts = load_contexts # (need a dict that has cont['monotonicity'])

    if intervention_type == 'change_insertions':
        # change one number -> change result
        # TODO change :)
        pass
    if intervention_type == "change_context_same_result":
        change_context_same_result(contexts, tokenizer)
        pass
    else:
        raise Exception('intervention_type not defined {}'.format(intervention_type))

    return interventions


def change_context_same_result():
    '''
    '''

    # fix insertion pair
    for _ in tqdm(range(args.examples_per_context)):

    interventions = []


    # for num contexts per template 
    # for 

    return interventions
