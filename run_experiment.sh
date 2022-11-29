# proj_root="${SCRATCH}/emnlp2022"
device="cuda"
model="roberta-large-mnli"
seed=0
random="not_random"
representation="words"
# data_path="${prj}/lm-intervention/data/mawps-asdiv-a_svamp/whole.csv" #should point to a svamp-formatted csv file
intervention_types="1b-2-3-4"  #possibilities: "0-1-1b-1c-2-3-4-10-11"
examples_per_context=50
out_dir="${proj_root}/out/${prompt}_${representation}_${random}_seed${seed}/maxn${max_n}/examplesn_${examples_per_template}"
wandb_mode="disabled"
transformers_cache_dir="${SCRATCH}/transformers_cache"
path_to_dict="${proj_root}/data/verbs-dictionaries.csv"

python ./experiments/estimate_effects.py $model $device $out_dir $random $representation $seed $prompt \
                                            $data_path $max_n $wandb_mode $intervention_types $examples_per_template \
                                            $transformers_cache_dir $path_to_dict
