# generate code-switched samples for manual inspection
# in analysis/results/code_switched, 80% of tweets has >= 50% English
def gen_code_switched_samples(df):
	df_code_switch = df_all[df_all["label"] == "code-switched"]
	return df_code_switch