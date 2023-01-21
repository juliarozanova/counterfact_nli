from nli_xy.encoding import build_dataset, load_tokenizer
from nli_xy.datasets import NLI_XY_Dataset
from interventions import Intervention
from typing import List
from tqdm import tqdm
import random
import os
from loguru import logger
import torch
from experiments.constants import DATA_PATH

# INTERVENTION_TYPES_TWO_RESULTS = ['0', '1', '1b', '1c', '4']
# INTERVENTION_TYPES_SINGLE_RESULT = ['2', '3', '10', '11']
# INTERVENTION_TYPES = ['change_insertions', 'change_context_same_result', 'change_context_different_result']

def construct_intervention_prompts(intervention_type, encode_config, args, sample_size=20):
    '''
    Create model agnostic text representations
    '''
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(encode_config)

    dataset = build_dataset(DATA_PATH, encode_config, tokenizer)

    if intervention_type == '0':
        # DCE (Same ground truth)
        logger.info('Constructing input intervention: change insertion, same result')
        interventions = change_insertions_interventions(dataset, tokenizer, sample_size, change_result=False)

    if intervention_type == '1':
        # TCE (Not constant ground truth )
        logger.info('Constructing input intervention: change insertion, change result')
        interventions = change_insertions_interventions(dataset, tokenizer, sample_size, change_result=True)

    if intervention_type == '2':
        # DSE (S -> R)
        logger.info('Constructing input intervention: change context (same monotonicity), same result')
        interventions = change_context_interventions(dataset, tokenizer, sample_size, change_monotonicity=False, change_result=False)

    if intervention_type == '3':
        #TCE (T->R)
        logger.info('Constructing input intervention: change context AND monotonicity, change result')
        interventions = change_context_interventions(dataset, tokenizer, sample_size, change_monotonicity=True, change_result=True)

    # TODO: An estimated DCE (O -> R) with minimal template edits?
    # if intervention_type == '4':

    # else:
    #     raise Exception(f'intervention_type not defined {intervention_type}')

    return interventions


def change_insertions_interventions(dataset: NLI_XY_Dataset, tokenizer, sample_size=20, change_result=True) -> List[Intervention]:
	interventions = []
	meta_df = dataset.meta_df.reset_index()
	context_groups = list(meta_df.groupby(by='context'))
	random.shuffle(context_groups)
	sampled_context_groups = context_groups[:sample_size]
	# sample_groups = np.arange(context_groups.ngroups)
	# np.random.shuffle()

	for context_text, context_group in tqdm(sampled_context_groups):
		insertion_groups = list(context_group.groupby(by='insertion_pair'))
		random.shuffle(insertion_groups)
		sampled_insertion_groups = insertion_groups[:sample_size]

		for insertion_pair_base, insertion_subgroup in sampled_insertion_groups:
			# Build base example
			insertion_subgroup = ensure_one_row(insertion_subgroup)

			# get row integer index value for base example
			row_id_base = insertion_subgroup.index.tolist()[0]

			premise_base = meta_df.at[row_id_base, 'premise']
			hypothesis_base = meta_df.at[row_id_base, 'hypothesis']

			input_toks_base = tokenizer.encode(
				premise_base,
				hypothesis_base
			)

			gold_label_base = insertion_subgroup.at[row_id_base, 'gold_label']

			# filter insertion groups by result
			if change_result:
				filtered_insertion_groups = context_group.loc[context_group.gold_label!=gold_label_base]
			elif not change_result: 
				filtered_insertion_groups = context_group.loc[context_group.gold_label==gold_label_base]
			
			filtered_insertion_groups = filtered_insertion_groups.groupby(by='insertion_pair')

			for insertion_pair_alt, insertion_subgroup_alt in filtered_insertion_groups:
				if insertion_pair_alt == insertion_pair_base:
					pass
				else:
					insertion_subgroup_alt = ensure_one_row(insertion_subgroup_alt)

					# get row integer index value
					row_id_alt = insertion_subgroup_alt.index.tolist()[0]
					premise_alt = meta_df.at[row_id_alt, 'premise']
					hypothesis_alt = meta_df.at[row_id_alt, 'hypothesis']


					input_toks_alt = tokenizer.encode(
						premise_alt,
						hypothesis_alt,
					)
					gold_label_alt = insertion_subgroup_alt.at[row_id_alt, 'gold_label']

					if change_result:
						assert gold_label_alt!=gold_label_base
					elif not change_result:
						assert gold_label_base==gold_label_alt

					intervention = Intervention(
						input_toks_base=input_toks_base,
						input_toks_alt=input_toks_alt,
						gold_label_base=gold_label_base,
						gold_label_alt=gold_label_alt,
						premise_base = premise_base,
						hypothesis_base = hypothesis_base,
						premise_alt = premise_alt,
						hypothesis_alt = hypothesis_alt
						# context_base = ,
					)

					interventions.append(intervention)

	return interventions

def change_context_interventions(dataset: NLI_XY_Dataset, tokenizer, sample_size=20, change_monotonicity=True, change_result=True) -> List[Intervention]:
	interventions = []
	meta_df = dataset.meta_df.reset_index()

	insertion_groups = list(meta_df.groupby(by='insertion_pair'))
	random.shuffle(insertion_groups)
	sampled_insertion_groups = insertion_groups[:sample_size]

	for insertion_pair, insertion_group in tqdm(sampled_insertion_groups):
		context_groups = list(insertion_group.groupby(by='context'))
		random.shuffle(context_groups)
		sampled_context_groups = context_groups[:sample_size]

		# iterate over single rows 
		for context_base, context_subgroup in sampled_context_groups:
			# Fix a base example
			context_subgroup = ensure_one_row(context_subgroup)

			# get row integer index value for base example
			row_id_base = context_subgroup.index.tolist()[0]
			context_monotonicity_base = context_subgroup.at[row_id_base, 'context_monotonicity']

			premise_base = meta_df.at[row_id_base, 'premise']
			hypothesis_base = meta_df.at[row_id_base, 'hypothesis']

			input_toks_base = tokenizer.encode(
				premise_base,
				hypothesis_base
			)

			gold_label_base = context_subgroup.at[row_id_base, 'gold_label']

			# get all rows for this insertion that have same/different gold label

			if change_result:
				filtered_context_group = insertion_group.loc[insertion_group.gold_label!=gold_label_base ]
			elif not change_result and not change_monotonicity: 
				filtered_context_group = insertion_group.loc[insertion_group.gold_label==gold_label_base]
			else:
				raise ValueError('Unexpected intervention configuration, Stopping')

			if change_monotonicity:
				filtered_context_group=filtered_context_group.loc[filtered_context_group.context_monotonicity!=context_monotonicity_base]
			elif not change_monotonicity:
				filtered_context_group=filtered_context_group.loc[filtered_context_group.context_monotonicity==context_monotonicity_base]
			else:
				raise ValueError('Unexpected intervention configuration, Stopping')


			filtered_context_groups = filtered_context_group.groupby(by='context')

			for context_alt, context_subgroup_alt in filtered_context_groups:
				if context_alt == context_base:
					pass
				else:
					context_subgroup_alt = ensure_one_row(context_subgroup_alt)

					# get row integer index value
					row_id_alt = context_subgroup_alt.index.tolist()[0]
					context_monotonicity_alt = context_subgroup_alt.at[row_id_alt, 'context_monotonicity']
					premise_alt = meta_df.at[row_id_alt, 'premise']
					hypothesis_alt = meta_df.at[row_id_alt, 'hypothesis']

					input_toks_alt = tokenizer.encode(
						premise_alt,
						hypothesis_alt,
					)
					gold_label_alt = context_subgroup_alt.at[row_id_alt, 'gold_label']

					if change_monotonicity:
						assert context_monotonicity_alt!=context_monotonicity_base
					if not change_monotonicity:
						assert context_monotonicity_alt==context_monotonicity_base
					if change_result:
						assert gold_label_alt!=gold_label_base
					elif not change_result:
						assert gold_label_alt==gold_label_base

					intervention = Intervention(
						input_toks_base=input_toks_base,
						input_toks_alt=input_toks_alt,
						gold_label_base=gold_label_base,
						gold_label_alt=gold_label_alt,
						premise_base = premise_base,
						hypothesis_base = hypothesis_base,
						premise_alt = premise_alt,
						hypothesis_alt = hypothesis_alt
						# context_base = ,
					)

					interventions.append(intervention)

	return interventions

def ensure_one_row(sub_df):
	try:
		assert  sub_df.shape[0]==1
	except AssertionError:
		# ensure repeats are dropped
			sub_df = sub_df.head(1)
	return sub_df
	