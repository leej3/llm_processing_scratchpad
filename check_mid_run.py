import pickle; import pandas as pd; from pathlib import Path
from sklearn.model_selection import train_test_split
output_filepath = Path("tempdata/llm_extractions/openai-gpt-4o-mini_20240926_162835.feather")
df = pd.read_feather("tempdata/combined_metadata.feather")
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

outputs = pickle.loads(output_filepath.with_suffix(".pkl").read_bytes())
df_llm = pd.DataFrame(outputs).set_index('idx').assign(reasoning_steps=lambda x: x["reasoning_steps"].astype(str))
df_out = df_llm.rename(columns={col: f"llm_{col}" for col in df_llm.columns}).join(df,how="left")
df_out.to_feather(output_filepath.with_suffix(".intermediate.feather"))
(
    # save the mismatched predictions to a tsv
    df_out
    .query("manual_is_open_data != llm_is_open_data")
    [["manual_is_open_data","llm_is_open_data","llm_data_sharing_statement","manual_data_statements","doi","filename","llm_reasoning_steps"]]
    .to_csv(str(output_filepath).replace(".feather","_mid_run_misses.tsv"),sep="\t",index=False)
)