from nli_xy.encoding import parse_encode_config, load_tokenizer
from interventions.intervention import construct_intervention_prompts
from intervention_models.intervention_model import Model
from result_utils import compute_aggregate_metrics, process_intervention_results, process_intervention_results_gpt3, compute_aggregate_metrics_for_col
from experiments.constants import ENCODE_CONFIG_FILE, DATA_PATH
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
        # args.model 

        # move config args to argparse object
        args.model = encode_config.representation_name

        model = Model()
        break