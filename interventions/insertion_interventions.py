from nli_xy.datasets import NLI_XY_Dataset
from interventions import Intervention
from typing import List
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from loguru import logger

def change_insertions_interventions(dataset: NLI_XY_Dataset, tokenizer, change_result=True, sample_size=20) -> List[Intervention]:
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

			res_base_string = insertion_subgroup.at[row_id_base, 'gold_label']

			# filter insertion groups by result
			if change_result:
				filtered_insertion_groups = context_group.loc[context_group.gold_label!=res_base_string]
				# filtered_insertion_groups = insertion_groups.filter(lambda x: x.gold_label!=res_base)
			elif not change_result: 
				filtered_insertion_groups = context_group.loc[context_group.gold_label==res_base_string]
				# filtered_insertion_groups = insertion_groups.filter(lambda x: x.gold_label==base_res)
			
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
					res_alt_string = insertion_subgroup_alt.at[row_id_alt, 'gold_label']

					if change_result:
						assert res_alt_string!=res_base_string
					elif not change_result:
						assert res_alt_string==res_base_string

					intervention = Intervention(
						input_toks_base=input_toks_base,
						input_toks_alt=input_toks_alt,
						res_base_string=res_base_string,
						res_alt_string=res_alt_string,
						premise_base = premise_base,
						hypothesis_base = hypothesis_base,
						premise_alt = premise_alt,
						hypothesis_alt = hypothesis_alt
						# context_base = ,
					)

					interventions.append(intervention)

	return interventions

def change_context_interventions(dataset: NLI_XY_Dataset, tokenizer, change_monotonicity=True, change_result=True, sample_size=20) -> List[Intervention]:
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

			res_base_string = context_subgroup.at[row_id_base, 'gold_label']

			# get all rows for this insertion that have same/different gold label

			if change_result:
				filtered_context_group = insertion_group.loc[insertion_group.gold_label!=res_base_string ]
			elif not change_result and not change_monotonicity: 
				filtered_context_group = insertion_group.loc[insertion_group.gold_label==res_base_string]
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
					res_alt_string = context_subgroup_alt.at[row_id_alt, 'gold_label']

					if change_monotonicity:
						assert context_monotonicity_alt!=context_monotonicity_base
					if not change_monotonicity:
						assert context_monotonicity_alt==context_monotonicity_base
					if change_result:
						assert res_alt_string!=res_base_string
					elif not change_result:
						assert res_alt_string==res_base_string

					intervention = Intervention(
						input_toks_base=input_toks_base,
						input_toks_alt=input_toks_alt,
						res_base_string=res_base_string,
						res_alt_string=res_alt_string,
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
	