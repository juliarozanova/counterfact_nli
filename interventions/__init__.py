#TODO Ensure intervention class doesn't rely on base_tok etc being mask tokens?

import torch
from nli_xy.analysis.eval_on_nli_task import three_class_models, two_class_models
from transformers import GPT2Tokenizer, BertTokenizer


class Intervention():
    '''
    Wrapper for all the possible interventions
    '''
    def __init__(self,
                 input_toks_base,
                 input_toks_alt,
                 res_base_string,
                 res_alt_string,
                 premise_base,
                 hypothesis_base,
                 premise_alt,
                 hypothesis_alt,
                 device='cpu'):
        ''' Stores metadata for two entailment examples which jointly constitute an "intervention".

        Parameters
        ----------
        input_toks_base : _type_
            _description_
        input_toks_alt : _type_
            _description_
        res_base_string : _type_
            _description_
        res_alt_string : _type_
            _description_
        premise_base : _type_
            _description_
        hypothesis_base : _type_
            _description_
        premise_alt : _type_
            _description_
        hypothesis_alt : _type_
            _description_
        device : str, optional
            _description_, by default 'cpu'

        Raises
        ------
        ValueError
            _description_
        '''
        super()
        self.device = device
        self.input_toks_base = input_toks_base
        self.input_toks_alt = input_toks_alt
        self.res_base_string = res_base_string
        self.res_alt_string = res_alt_string
        self.premise_base = premise_base
        self.hypothesis_base = hypothesis_base
        self.premise_alt = premise_alt
        self.hypothesis_alt = hypothesis_alt

        self.input_toks_base = torch.LongTensor(self.input_toks_base).to(device)
        self.input_toks_alt = torch.LongTensor(self.input_toks_alt).to(device)