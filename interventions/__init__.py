#TODO Ensure intervention class doesn't rely on base_tok etc being mask tokens?

import torch
from transformers import GPT2Tokenizer, BertTokenizer


class Intervention():
    '''
    Wrapper for all the possible interventions
    '''
    def __init__(self,
                 base_input_toks,
                 alt_input_toks,
                 base_res,
                 alt_res,
                 multitoken=False,
                 device='cpu'):
        super()
        self.device = device
        self.multitoken = multitoken
        self.base_input_toks = base_input_toks
        self.alt_input_toks = alt_input_toks
        self.base_res = base_res
        self.alt_res = alt_res

        self.base_input_toks = torch.LongTensor(self.base_input_toks).to(device)
        self.alt_input_toks = torch.LongTensor(self.alt_input_toks).to(device)


    def set_results(self, res_base, res_alt):
        '''
        Originally: tokenize the result number as a word
        '''
        self.res_base_string  = res_base
        self.res_alt_string = res_alt

        if self.enc is not None:
            # 'a ' added to input so that tokenizer understands that first word
            # follows a space.
            self.res_base_tok = self.enc.tokenize('a ' + res_base)[1:]
            self.res_alt_tok = self.enc.tokenize('a ' + res_alt)[1:]
            if not self.multitoken:
                assert len(self.res_base_tok) == 1, '{} - {}'.format(self.res_base_tok, res_base)
                assert len(self.res_alt_tok) == 1, '{} - {}'.format(self.res_alt_tok, res_alt)

            self.res_base_tok = self.enc.convert_tokens_to_ids(self.res_base_tok)
            self.res_alt_tok = self.enc.convert_tokens_to_ids(self.res_alt_tok)

    def set_result(self, res):
        self.res_string = res

        if self.enc is not None:
            self.res_tok = self.enc.tokenize('a ' + res)[1:]
            if not self.multitoken:
                assert (len(self.res_tok) == 1)