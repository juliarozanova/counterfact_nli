from nli_xy.analysis.eval_on_nli_task import relabel_three_class_predictions, three_class_models, two_class_models
from interventions import Intervention
import pandas as pd
import numpy as np
from loguru import logger
from torch import softmax

def batch(iterable, bsize=1):
    total_len = len(iterable)
    for ndx in range(0, total_len, bsize):
        yield list(iterable[ndx:min(ndx + bsize, total_len)])

def kl_div(a,b):
    return sum([a[i] * np.log(a[i] / b[i]) if a[i] > 0 else 0 for i in range(len(a))])

def tv_distance(a,b):
    return np.sum(np.abs(a - b)) / 2

def d_inf(a,b):
    return np.max(np.log(np.maximum(a/b,b/a)))

# def softmax(x):
#     return np.exp(x) / sum(np.exp(x))


def compute_relative_confidence_change(distrib_base, distrib_alt, c1, c2):
    candidate1_base_prob = distrib_base[c1]
    candidate2_base_prob = distrib_base[c2]
    candidate1_alt1_prob = distrib_alt[c1]
    candidate2_alt1_prob = distrib_alt[c2]
    base_error = candidate2_base_prob / candidate1_base_prob
    alt_error = candidate1_alt1_prob / candidate2_alt1_prob
    total_effect = 1 / (alt_error * base_error) - 1

    return  total_effect


class InterventionResult():
    def __init__(self, intervention: Intervention,
                        logits_base,
                        logits_alt,
                        probs_base,
                        probs_alt,
                        model_name,
                        ):

        self.logits_base = logits_base
        self.logits_alt = logits_alt
        self.probs_base = probs_base
        self.probs_alt = probs_alt
        self.model_name = model_name

        self.res_base_string = intervention.res_base_string
        self.res_alt_string = intervention.res_alt_string


        self.interpret_gold_labels_as_ints()


    def interpret_gold_labels_as_ints(self):
        '''
            Creates two values, the "res" (gold expected result, which is dependent on the shape of model outputs
                and maps the gold label to the relevant model output label)
            and the "y_true" (independent of model: integer categorization of gold label)

        Raises
        ------
        ValueError
            If it is unknown how the model's outputs should be interpreted in terms of entailment and non-entailment
        '''

        # Interpret res_base_string as a value according the model's predictions shape
        # TODO should we return 1 or zero? Non-entailment is more like neutral than contradiction, check we're not labelling as contra
        # TODO Alternatively: consider the union probability of classes 0 and 1 as a new normalized output?
        #   then we would not need to destinguish between "res" and "y_true"

        if self.model_name in three_class_models:
            if self.res_base_string == 'entailment':
                self.res_base = 2
            if self.res_base_string == 'non-entailment':
                self.res_base = 1
            if self.res_alt_string == 'entailment':
                self.res_alt = 2
            if self.res_alt_string == 'non-entailment':
                self.res_alt = 1
        elif self.model_name in two_class_models: 
            if self.res_base_string == 'entailment':
                self.res_base = 1 
            if self.res_base_string == 'non-entailment':
                self.res_base = 0
            if self.res_alt_string == 'entailment':
                self.res_alt = 1
            if self.res_alt_string == 'non-entailment':
                self.res_alt = 0
        else:
            # todo: command line 
            raise ValueError(f'Model name {self.model_name} not recognised. Is it two or three class entailment model?')

        if self.res_base_string == 'entailment':
            self.y_true_base = 1
        if self.res_base_string == 'non-entailment':
            self.y_true_base = 0
        if self.res_alt_string == 'entailment':
            self.y_true_alt = 1
        if self.res_alt_string == 'non-entailment':
            self.y_true_alt = 0


def process_intervention_results(interventions, intervention_results, representation):
    results = []
    for example in intervention_results:
        intervention_result = intervention_results[example]
        intervention = interventions[example]

        normalized_base = np.array(softmax(intervention_result.logits_base, dim=0))
        normalized_alt = np.array(softmax(intervention_result.logits_alt, dim=0))

        js_div = (kl_div(normalized_base, (normalized_alt + normalized_base) / 2) + kl_div(normalized_alt, (normalized_alt + normalized_base) / 2)) / 2
        tv_norm = tv_distance(normalized_base, normalized_alt)
        l_inf_div = d_inf(normalized_base, normalized_alt)

        # Causal effect
        pred_base = normalized_base.argmax()
        pred_alt = normalized_alt.argmax()
        # TODO: do we want to do this with pred (in 3-sapce) or with y_pred (in 2-space)?
        causal_effect = np.abs(pred_base - pred_alt)

        if intervention_result.model_name in three_class_models:
            y_pred_base = relabel_three_class_predictions(pred_base)
            y_pred_alt = relabel_three_class_predictions(pred_alt)

        elif intervention_result.model_name in two_class_models:
            y_pred_base = pred_base
            y_pred_alt = pred_alt
            try:
                assert y_pred_base not in [2,'2']
            except AssertionError:
                logger.warning(f'Are you sure {intervention_result.model_name} is a two class model? It is predicting class {pred_base} and {pred_alt}!')


        res_base = intervention_result.res_base
        res_alt = intervention_result.res_alt


        is_correct_base = (1 if y_pred_base==intervention_result.y_true_base else 0)
        is_correct_alt = (1 if y_pred_alt==intervention_result.y_true_alt else 0)

        metric_dict = {
            'example': example,
            'res_base_string': intervention_result.res_base_string,
            'res_alt_string': intervention_result.res_alt_string,
            'res_base': res_base,
            'res_alt': res_alt,
            'pred_base': pred_base,
            'pred_alt': pred_alt,
            'y_true_base': intervention_result.y_true_base,
            'y_true_alt': intervention_result.y_true_alt,
            'y_pred_base': y_pred_base,
            'y_pred_alt': y_pred_alt,
            'is_correct_base': is_correct_base,
            'is_correct_alt': is_correct_alt,
            'premise_base': intervention.premise_base,
            'hypothesis_base': intervention.hypothesis_base,
            'premise_alt': intervention.premise_alt,
            'hypothesis_alt': intervention.hypothesis_alt,
            # 'context_monotonicity_base', 
            # 'context_monotonicity_alt',
            # 'insertion_rel_base',
            # 'insertion_rel_alt',
            'distrib_base': normalized_base,
            'distrib_alt': normalized_alt,
            'js_div': js_div,
            'tv_norm': tv_norm,
            'l_inf_div': l_inf_div,
            'causal_effect': causal_effect
        }


        # TODO:  Adjust normalized_base according to the model, and create a standardised one with proba("non-entailment") and proba("entailment")?

        prob_base_res_base = normalized_base[res_base]
        prob_base_res_alt = normalized_base[res_alt]
        prob_alt_res_base = normalized_alt[res_base]
        prob_alt_res_alt = normalized_alt[res_alt]

        # Confidence change
        base_confidence_change = (prob_base_res_base - prob_alt_res_base) / prob_alt_res_base
        alt_confidence_change = (prob_alt_res_alt - prob_base_res_alt) / prob_base_res_alt
        confidence_change = (base_confidence_change + alt_confidence_change) / 2

        # Absolute value confidence change
        abs_base_confidence_change = np.abs(prob_base_res_base - prob_alt_res_base) / prob_alt_res_base
        abs_alt_confidence_change = np.abs(prob_alt_res_alt - prob_base_res_alt) / prob_base_res_alt
        abs_confidence_change = (abs_base_confidence_change + abs_alt_confidence_change) / 2

        # relative confidence change
        relative_confidence_change = compute_relative_confidence_change(normalized_base, normalized_alt, res_base, res_alt)
        metric_dict['relative_confidence_change'] = relative_confidence_change

        # error change
        error_base = np.abs(pred_base - res_base)
        error_alt = np.abs(pred_alt - res_alt)
        error_change = np.abs(error_base - error_alt)

        metric_dict['error_change'] = error_change
        metric_dict['confidence_change'] = confidence_change
        metric_dict['abs_confidence_change'] = abs_confidence_change

        results.append(metric_dict)

    return pd.DataFrame(results)


def compute_aggregate_metrics_for_col(col):
    return {'mean' : col.mean(), 'sem' : col.sem(), 'std' : col.std()}

def compute_aggregate_metrics(df, single_result):
    metrics_dict = {}
    measures = ['is_correct_base', 'is_correct_alt', 'causal_effect', 'js_div', 'tv_norm', 'error_change', 'confidence_change', 'abs_confidence_change']
    if not single_result:
        measures = measures + ['relative_confidence_change']

    for measure in measures:
        metric_dict = compute_aggregate_metrics_for_col(df[measure])
        metrics_dict[measure] = metric_dict

    return metrics_dict