from nli_xy.encoding import parse_encode_config
from interventions.interventions import construct_intervention_prompts
from interventions.intervention_model import Model
from interventions.result_utils import compute_aggregate_metrics, process_intervention_results
from experiments.constants import ENCODE_CONFIG_FILE
from loguru import logger
import pandas as pd
import os
import argparse

if __name__=="__main__":

    # if not (len(sys.argv) == 16):
    #     print("USAGE: python ", sys.argv[0], 
    #             "<model> <device> <out_dir> <random_weights> <representation> <seed> <prompt> <path_to_num_data> <examples> <wandb_mode> <intervention_type> <examples_per_template> <transformers_cache_dir> <path_to_dict>")

    PARAMETERS = {
        'seed' : 3,  # to allow consistent sampling
    }

    intervention_types = ['0', '1', '2', '3']
    RESULTS_DIR='experiments/results/'

    args = argparse.Namespace(**PARAMETERS)

    encode_configs = parse_encode_config(ENCODE_CONFIG_FILE)
    # change representations to model_name?
    for rep_name, encode_config in encode_configs["representations"].items():
        logger.info(f'Performing interventions for the model {rep_name}:')
        model = Model(
            device=encode_config['device'],
            model_name=rep_name,
            model_version=encode_config['encoder_model'],
        )

        for intervention_type in intervention_types:
            interventions = construct_intervention_prompts(intervention_type, encode_config, args, sample_size=20)
            intervention_results = model.intervention_experiment(interventions)

            results_df = process_intervention_results(intervention_results)
            metrics_dict = compute_aggregate_metrics(results_df, single_result=False)


            INTERVENTION_RESULTS_DIR = os.path.join(RESULTS_DIR, f'intervention_{intervention_type}/{rep_name}/')
            if not os.path.exists(INTERVENTION_RESULTS_DIR):
                os.makedirs(INTERVENTION_RESULTS_DIR)

            with open(os.path.join(INTERVENTION_RESULTS_DIR, f'meta.tsv'), 'w+') as file:
                # TODO: write with the 'mean', 'sem' and 'std' as index columns?
                file.write(results_df.to_csv(sep='\t', index=False))

            with open(os.path.join(INTERVENTION_RESULTS_DIR, f'summary_results.tsv'), 'w+') as file:
                file.write(pd.DataFrame(metrics_dict).to_csv(sep='\t'))