import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
import statistics
from loguru import logger
from experiments.utils import get_label_maps

from interventions.result_utils import InterventionResult
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

        self.device = device


        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_version,
            output_attentions=output_attentions,
            cache_dir = transformers_cache_dir
        )
        self.label2id, self.id2label = get_label_maps(self.model, self.model_name)

        self.model.eval()
        self.model.to(device)


        if random_weights:
            logger.info('Randomizing weights')
            self.model.init_weights()


    def set_st_ids(self, tokenizer):
        # Special token id's: (mask, cls, sep)
        self.st_ids = (tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id)


    def intervention_experiment(self, interventions):

        # intervention_results = {}
        intervention_results = []
        # for idx, intervention in enumerate(tqdm(interventions, desc='Running model predictions on interventions:')):
        for intervention in tqdm(interventions, desc='Running model predictions on interventions'):
            intervention_results.append(self.intervention_full_ditribution_experiment(intervention))

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
                                    model_name=self.model_name,
                                    label2id=self.label2id,
                                    id2label=self.id2label,
                                    # pred_base,
                                    # pred_alt
                                    )


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
