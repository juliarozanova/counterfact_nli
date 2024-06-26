import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
import statistics
from loguru import logger
from result_utils import InterventionResult
from transformers import (
    BertForSequenceClassification, RobertaForSequenceClassification,
    AutoModelForSequenceClassification
)


class Model():

    def __init__(self,
                 device='cpu',
                 model_name='',
                 output_attentions=False,
                 random_weights=False,
                 model_version='gpt2',
                 transformers_cache_dir=None):

        super()

        self.model_name = model_name
        self.is_gpt2 = (model_version.startswith('gpt2') or model_version.startswith('distilgpt2'))
        self.is_gptj = model_version.startswith('EleutherAI/gpt-j')
        self.is_bert = model_version.startswith('bert')
        self.is_roberta = model_version.startswith('roberta')
        self.is_gptneo = model_version.startswith('EleutherAI/gpt-neo')
        self.is_gpt3 = model_version.startswith('gpt3')
        # assert (self.is_gpt2 or self.is_bert or self.is_roberta or self.is_gptj or self.is_gptneo)

        self.device = device

        model_class = (BertForSequenceClassification if self.is_bert else
                  RobertaForSequenceClassification if self.is_roberta else
                  AutoModelForSequenceClassification)

        self.model = model_class.from_pretrained(
            model_version,
            output_attentions=output_attentions,
            cache_dir = transformers_cache_dir
        )
        self.model.eval()
        self.model.to(device)

        if random_weights:
            logger.info('Randomizing weights')
            self.model.init_weights()


    def set_st_ids(self, tokenizer):
        # Special token id's: (mask, cls, sep)
        self.st_ids = (tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id)

    def set_vocab_subset(self, tokenizer, representation, max_n):
        if representation == 'arabic':
            self.vocab_subset = [tokenizer.encode('a ' + str(i))[1:] for i in range(max_n + 1)]
        # elif representation == 'words':
        #     self.vocab_subset = [tokenizer.encode('a ' + convert_to_words(str(i)))[1:] for i in range(max_n + 1)]
        # else:
            raise Exception('Representation unknown: {}'.format(representation))


    def intervention_experiment(self, interventions):

        intervention_results = {}
        for idx, intervention in enumerate(tqdm(interventions, desc='performing interventions')):
            intervention_results[idx] = self.intervention_full_ditribution_experiment(intervention)

        return intervention_results


    # TODO make sure logits are coming from entailment prediction, not mask token or something
    def intervention_full_ditribution_experiment(self, intervention):
        base_tok = intervention.input_toks_base.unsqueeze(0)
        alt_tok = intervention.input_toks_alt.unsqueeze(0)
        logits_base, probs_base = self.get_distribution_for_examples(base_tok)
        logits_alt, probs_alt = self.get_distribution_for_examples(alt_tok)

        return InterventionResult(intervention,
                                    logits_base,
                                    logits_alt,
                                    probs_base,
                                    probs_alt,
                                    model_name=self.model_name
                                    # pred_base,
                                    # pred_alt
                                    )

    def get_top_k_logprobs(self, context, k=100):
        with torch.no_grad():
            logits = self.model(context.to(self.device))[0].to('cpu')
            logits = logits[:, -1, :]
            probs = F.log_softmax(logits, dim=-1)
            probs = probs.squeeze().tolist()

        probs_pairs = [ (i , p) for i, p in enumerate(probs)]
        sorted_probs = sorted(probs_pairs, key= lambda item: item[1], reverse=True)
        probs_dict = dict(sorted_probs[:k])
        return probs_dict


    def get_distribution_for_examples(self, context):
        with torch.no_grad():
            logits = self.model(context.to(self.device))[0].to('cpu')
            # logits = logits[:, -1, :]
            # logits = logits[:, -1]
            probs = F.softmax(logits, dim=-1)

            logits = logits.squeeze()
            probs = probs.squeeze()
            # logits_subset = logits[:, self.vocab_subset].squeeze().tolist()
            # probs_subset = probs[:, self.vocab_subset].squeeze().tolist()

        return logits, probs


    def get_distribution_multitoken(self, context):
        mean_probs = []
        mean_logits = []
        with torch.no_grad():
            for candidate in self.vocab_subset:
                torch.cuda.empty_cache()
                token_log_probs = []
                token_logits = []
                combined = context.squeeze().tolist() + candidate
                # Exclude last token position when predicting next token
                batch = torch.tensor(combined[:-1]).unsqueeze(dim=0).to(self.device)
                # Shape (batch_size, seq_len, vocab_size)
                logits = self.model(batch)[0].to('cpu')
                # Shape (seq_len, vocab_size)
                logits = logits.squeeze()
                log_probs = F.log_softmax(logits, dim=-1)
                context_end_pos = len(context.squeeze().tolist()) - 1
                continuation_end_pos = context_end_pos + len(candidate)
                # TODO: Vectorize this
                # Up to but not including last token position
                for i in range(context_end_pos, continuation_end_pos):
                    next_token_id = combined[i + 1]
                    next_token_log_prob = log_probs[i][next_token_id].item()
                    token_log_probs.append(next_token_log_prob)
                    next_token_logit = logits[i][next_token_id].item()
                    token_logits.append(next_token_logit)
                mean_token_log_prob = statistics.mean(token_log_probs)
                mean_token_prob = math.exp(mean_token_log_prob)
                mean_probs.append(mean_token_prob)
                mean_logit = statistics.mean(token_logits)
                mean_logits.append(mean_logit)
        return mean_logits, mean_probs