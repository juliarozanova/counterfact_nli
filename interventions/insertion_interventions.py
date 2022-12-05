from nli_xy.datasets import NLI_XY_Dataset
from interventions import Intervention
from typing import List
from tqdm import tqdm
import pandas as pd
from loguru import logger

def change_insertions_interventions_same_result(dataset: NLI_XY_Dataset, tokenizer, change_result=True) -> List[Intervention]:
	interventions = []
	meta_df = dataset.meta_df.reset_index()
	context_groups = meta_df.groupby(by='context')

	for context_text, context_group in tqdm(context_groups):
		insertion_groups = context_group.groupby(by='insertion_pair')

		for insertion_pair, insertion_subgroup in insertion_groups:
			insertion_subgroup = ensure_one_row(insertion_subgroup)

			# get row integer index value
			row_id = insertion_subgroup.index.tolist()[0]

			base_input_toks = tokenizer.encode(
				meta_df.at[row_id, 'premise'],
				meta_df.at[row_id, 'hypothesis'],
			)

			base_res = insertion_subgroup.at[row_id, 'gold_label']

			# filter insertion groups by result

			if change_result:
				filtered_insertion_groups = context_group.loc[context_group.gold_label!=base_res]
				# filtered_insertion_groups = insertion_groups.filter(lambda x: x.gold_label!=base_res)
			else: 
				filtered_insertion_groups = context_group.loc[context_group.gold_label==base_res]
				# filtered_insertion_groups = insertion_groups.filter(lambda x: x.gold_label==base_res)
			
			filtered_insertion_groups = filtered_insertion_groups.groupby(by='insertion_pair')

			for alt_insertion_pair, alt_insertion_subgroup in filtered_insertion_groups:
				if alt_insertion_pair == insertion_pair:
					pass
				else:
					alt_insertion_subgroup = ensure_one_row(alt_insertion_subgroup)

					# get row integer index value
					alt_row_id = alt_insertion_subgroup.index.tolist()[0]

					alt_input_toks = tokenizer.encode(
						meta_df.at[alt_row_id, 'premise'],
						meta_df.at[alt_row_id, 'hypothesis'],
					)
					alt_res = alt_insertion_subgroup.at[alt_row_id, 'gold_label']

					if change_result:
						assert alt_res!=base_res
					elif not change_result:
						assert alt_res==base_res

					intervention = Intervention(
						base_input_toks=base_input_toks,
						alt_input_toks=alt_input_toks,
						base_res=base_res,
						alt_res=alt_res
					)

					interventions.append(intervention)
				break
		break

	return interventions

def ensure_one_row(sub_df):
	try:
		assert  sub_df.shape[0]==1
	except AssertionError:
		# ensure repeats are dropped
			sub_df = sub_df.head(1)
	return sub_df
	