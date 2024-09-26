
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

# df_llm = pd.DataFrame(pickle.loads(Path("temp.pklz").read_bytes())).set_index("idx")
# full_df = pd.read_feather("tempdata/combined_metadata.feather")
# train_df, test_df = train_test_split(full_df, test_size=0.1, random_state=42)
# df_out =  df_llm.rename(columns={col: f"llm_{col}" for col in df_llm.columns}).join(train_df)


import pandas as pd
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix
# df = pd.read_feather("tempdata/llm_extractions/openai-gpt-4o-mini-2024-07-18_20240925_144254.feather")[['manual_is_open_data', 'llm_is_open_data']].dropna().astype(bool) #'oddpub_is_open_data'
oddpub = (
    pd.read_csv("tempdata/oddpub.tsv", sep="\t")
    .rename(columns={"is_open_data":"oddpub_is_open_data"})
    .assign(filename= lambda df: df.article.str.split("/").str[-1].str.replace(".txt",".pdf"))
    [["filename","oddpub_is_open_data"]]
)
sonnet35 = (
    pd.read_feather("tempdata/llm_extractions/anthropic-claude-3.5-sonnet_20240926_161408.feather")
    .rename(columns={"llm_is_open_data":"sonnet35_is_open_data"})
    [["filename","sonnet35_is_open_data"]]
)
df = (
    pd.read_feather("tempdata/llm_extractions/openai-gpt-4o-mini_20240926_162835.feather")
    .rename(columns={"llm_is_open_data":"4o-mini_is_open_data"}).merge(
        sonnet35,on="filename",how="inner"
    )
    [['manual_is_open_data', 'sonnet35_is_open_data','4o-mini_is_open_data',"filename"]]
    .assign(filename = lambda df: df.filename.str.split("/").str[-1])
    .merge(oddpub,on="filename",how="inner")
    .dropna()
    .astype(bool)
)
# Assuming 'df' is the dataframe
# 'oddpub_is_open_data', 'llm_is_open_data', and 'manual_is_open_data' columns

categories = ['True Positive', 'True Negative', 'False Positive', 'False Negative']


def calculate_confusion_matrix_values(df, actual_col, predicted_col):
    cm = confusion_matrix(df[actual_col], df[predicted_col])
    true_positive = cm[0, 0]
    true_negative = cm[1, 1]
    false_positive = cm[1, 0]
    false_negative = cm[0, 1]
    values = [true_positive, true_negative, false_positive, false_negative]
    percentages = [f"{(v / sum(values)) * 100:.1f}%" for v in values]
    return values, percentages

# Confusion matrix for oddpub
oddpub_values, oddpub_percentages = calculate_confusion_matrix_values(df, 'manual_is_open_data', 'oddpub_is_open_data')

# Confusion matrix for LLM
plot_values = {
"sonnet35": calculate_confusion_matrix_values(df, 'manual_is_open_data', 'sonnet35_is_open_data'),
"4o-mini": calculate_confusion_matrix_values(df, 'manual_is_open_data', '4o-mini_is_open_data'),
"oddpub": calculate_confusion_matrix_values(df, 'manual_is_open_data', 'oddpub_is_open_data'),

}


# Calculate percentages for text display

# Create the bar chart
fig = go.Figure(data=[
    go.Bar(name=key, x=categories, y=val[0], text=val[1], textposition='auto') for key, val in plot_values.items()
    # go.Bar(name='LLM', x=categories, y=llm_values, text=llm_percentages, textposition='auto', marker_color='lightblue')

])

# Update layout
fig.update_layout(
    title='Comparison of oddpub and LLM with Human Labelled Ground Truth',
    xaxis_title='Categories',
    yaxis_title='Count',
    barmode='group',
)

# Show the figure
fig.write_html("llm_performance_comparison.html")
fig.show()