from nli_xy.encoding import parse_encode_config
from interventions.intervention import construct_intervention_prompts
from intervention_models.intervention_model import Model
from result_utils import compute_aggregate_metrics, process_intervention_results, process_intervention_results_gpt3, compute_aggregate_metrics_for_col
from experiments.constants import ENCODE_CONFIG_FILE, DATA_PATH
from loguru import logger
import json
import argparse

if __name__=="__main__":

    # if not (len(sys.argv) == 16):
    #     print("USAGE: python ", sys.argv[0], 
    #             "<model> <device> <out_dir> <random_weights> <representation> <seed> <prompt> <path_to_num_data> <examples> <wandb_mode> <intervention_type> <examples_per_template> <transformers_cache_dir> <path_to_dict>")

    PARAMETERS = {
        'seed' : 2,  # to allow consistent sampling
    }
    intervention_type = '0'

    args = argparse.Namespace(**PARAMETERS)

    encode_configs = parse_encode_config(ENCODE_CONFIG_FILE)
    # change representations to model_name?
    for rep_name, encode_config in encode_configs["representations"].items():
        interventions = construct_intervention_prompts(intervention_type, encode_config, args)

        model = Model(
            device=encode_config['device'],
            model_version=encode_config['encoder_model']
        )
        intervention_results = model.intervention_experiment(interventions)

        results_df = process_intervention_results(interventions, intervention_results, args.max_n, args.representation, single_result=single_result)
        metrics_dict = compute_aggregate_metrics(results_df, single_result=False)

        print(json.dumps(metrics_dict, indent=4))
        # metrics_dict = { 'int{}_'.format(itype) + k 


        break