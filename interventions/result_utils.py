from experiments.utils import LABEL2ID_2CLASS
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
                        label2id,
                        id2label,
                        ):

        self.intervention = intervention
        self.logits_base = logits_base
        self.logits_alt = logits_alt
        self.probs_base = probs_base
        self.probs_alt = probs_alt
        self.model_name = model_name
        self.label2id = label2id
        self.id2label = id2label

        self.gold_label_base = intervention.gold_label_base
        self.gold_label_alt = intervention.gold_label_alt

        self.y_true_3_class_base, self.y_true_2_class_base = self.set_y_true_labels(self.gold_label_base)
        self.y_true_3_class_alt, self.y_true_2_class_alt = self.set_y_true_labels(self.gold_label_alt)

    def set_y_true_labels(self, gold_label):
        
        try:
            y_true_3_class = self.label2id[gold_label]
        except KeyError:
            if gold_label=="non-entailment":
                # hack: for 3 class models, assign the same label as the "neutral" class
                y_true_3_class = self.label2id["neutral"]

        # model agnostic 2 class label id
        y_true_2_class = LABEL2ID_2CLASS[gold_label]

        return y_true_3_class, y_true_2_class


def process_intervention_results(intervention_results):
    results = []
    for intervention_result in intervention_results:
        intervention = intervention_result.intervention

        normalized_base = np.array(softmax(intervention_result.logits_base, dim=0))
        normalized_alt = np.array(softmax(intervention_result.logits_alt, dim=0))

        js_div = (kl_div(normalized_base, (normalized_alt + normalized_base) / 2) + kl_div(normalized_alt, (normalized_alt + normalized_base) / 2)) / 2
        tv_norm = tv_distance(normalized_base, normalized_alt)
        l_inf_div = d_inf(normalized_base, normalized_alt)

        # Model dependent predictions
        pred_base = normalized_base.argmax()
        pred_alt = normalized_alt.argmax()


        # transform labels to unified 2 class output space ()
        y_pred_base = LABEL2ID_2CLASS[intervention_result.id2label[pred_base]]
        y_pred_alt = LABEL2ID_2CLASS[intervention_result.id2label[pred_alt]]

        # Causal effect
        causal_effect = np.abs(y_pred_base - y_pred_alt)

        y_true_3_class_base = intervention_result.y_true_3_class_base
        y_true_3_class_alt = intervention_result.y_true_3_class_alt
        y_true_2_class_base = intervention_result.y_true_2_class_base
        y_true_2_class_alt = intervention_result.y_true_2_class_alt


        is_correct_base = (1 if y_pred_base==intervention_result.y_true_2_class_base else 0)
        is_correct_alt = (1 if y_pred_alt==intervention_result.y_true_2_class_alt else 0)

        metric_dict = {
            'gold_label_base': intervention_result.gold_label_base,
            'gold_label_alt': intervention_result.gold_label_alt,
            'y_true_3_class_base': y_true_3_class_base,
            'y_true_3_class_alt': y_true_3_class_alt,
            'y_true_2_class_base': y_true_2_class_base,
            'y_true_2_class_alt': y_true_2_class_alt,
            'pred_base': pred_base,
            'pred_alt': pred_alt,
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
            'causal_effect': causal_effect,
        }

        # Metrics using 3 class results will not be comparable. Change or ignore.

        # TODO:  Adjust normalized_base according to the model, and create a standardised one with proba("non-entailment") and proba("entailment")?

        # 3 class confidence changes (both "normalized" and "y_true_3_class" are model dependant)
        prob_base_res_base = normalized_base[y_true_3_class_base]
        prob_base_res_alt = normalized_base[y_true_3_class_alt]
        prob_alt_res_base = normalized_alt[y_true_3_class_base]
        prob_alt_res_alt = normalized_alt[y_true_3_class_alt]

        # Confidence change
        base_confidence_change = (prob_base_res_base - prob_alt_res_base) / prob_alt_res_base
        alt_confidence_change = (prob_alt_res_alt - prob_base_res_alt) / prob_base_res_alt
        confidence_change = (base_confidence_change + alt_confidence_change) / 2

        # Absolute value confidence change
        abs_base_confidence_change = np.abs(prob_base_res_base - prob_alt_res_base) / prob_alt_res_base
        abs_alt_confidence_change = np.abs(prob_alt_res_alt - prob_base_res_alt) / prob_base_res_alt
        abs_confidence_change = (abs_base_confidence_change + abs_alt_confidence_change) / 2

        # relative confidence change
        relative_confidence_change = compute_relative_confidence_change(normalized_base, normalized_alt, y_true_3_class_base, y_true_3_class_alt)
        metric_dict['relative_confidence_change'] = relative_confidence_change

        # error change
        error_base = np.abs(pred_base - y_true_3_class_base)
        error_alt = np.abs(pred_alt - y_true_3_class_alt)
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
    measures = ['is_correct_base',
     'is_correct_alt',
    'causal_effect',
    'js_div',
    'tv_norm',
    'error_change', 
    'confidence_change', 
    'abs_confidence_change']
    if not single_result:
        measures = measures + ['relative_confidence_change']

    for measure in measures:
        metric_dict = compute_aggregate_metrics_for_col(df[measure])
        metrics_dict[measure] = metric_dict

    return metrics_dict