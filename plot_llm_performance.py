
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
# df = pd.read_feather("tempdata/llm_extractions/openai-gpt-4o-mini_20240926_120539.feather")
df = pd.read_feather("tempdata/llm_extractions/openai-gpt-4o-mini-2024-07-18_20240925_144254.feather")[['manual_is_open_data', 'llm_is_open_data']].dropna().astype(bool) #'oddpub_is_open_data'
# Assuming 'df' is the dataframe
# 'oddpub_is_open_data', 'llm_is_open_data', and 'manual_is_open_data' columns

categories = ['True Positive', 'True Negative', 'False Positive', 'False Negative']


# Confusion matrix for oddpub
# cm_oddpub = confusion_matrix(df['manual_is_open_data'], df['oddpub_is_open_data'], labels=[1, 0])
# # Extracting counts from confusion matrices
# true_positive_oddpub = cm_oddpub[0, 0]
# true_negative_oddpub = cm_oddpub[1, 1]
# false_positive_oddpub = cm_oddpub[1, 0]
# false_negative_oddpub = cm_oddpub[0, 1]
# oddpub_values = [true_positive_oddpub, true_negative_oddpub, false_positive_oddpub, false_negative_oddpub]
# oddpub_percentages = [f"{(v / sum(oddpub_values)) * 100:.1f}%" for v in oddpub_values]

# Confusion matrix for LLM
cm_llm = confusion_matrix(df['manual_is_open_data'], df['llm_is_open_data'])
true_positive_llm = cm_llm[0, 0]
true_negative_llm = cm_llm[1, 1]
false_positive_llm = cm_llm[1, 0]
false_negative_llm = cm_llm[0, 1]
llm_values = [true_positive_llm, true_negative_llm, false_positive_llm, false_negative_llm]
llm_percentages = [f"{(v / sum(llm_values)) * 100:.1f}%" for v in llm_values]


# Calculate percentages for text display

# Create the bar chart
fig = go.Figure(data=[
    # go.Bar(name='oddpub', x=categories, y=oddpub_values, text=oddpub_percentages, textposition='auto', marker_color='purple'),
    go.Bar(name='LLM', x=categories, y=llm_values, text=llm_percentages, textposition='auto')
])

# Update layout
fig.update_layout(
    title='Comparison of oddpub and LLM with Human Labelled Ground Truth',
    xaxis_title='Categories',
    yaxis_title='Count',
    barmode='group',
)

# Show the figure
fig.show()