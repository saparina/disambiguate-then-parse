import numpy as np
import pandas as pd
from dataset import add_nl_interpretations, merge_all_insert_statements

# Load the dataset
df = pd.read_csv("data/ambrosia/data/ambrosia.csv")

# Preprocess the data
df.loc[df['ambig_type'] == 'attachment', 'question'] += " Show them in one table."
df.loc[df['ambig_type'] == 'attachment', 'ambig_question'] += " Show them in one table."
df['db_dump_processed'] = df.apply(lambda row: merge_all_insert_statements(row['db_file'], row['db_dump']), axis=1)

# Add natural language interpretations
df = add_nl_interpretations(df)

# Get unique db_file values and shuffle them
unique_db_files = df['db_file'].unique()
np.random.seed(42)  # For reproducibility
np.random.shuffle(unique_db_files)

train_split_index = int(0.8 * len(unique_db_files))
val_split_index = int(0.9 * len(unique_db_files))

train_db_files = unique_db_files[:train_split_index]
test_db_files = unique_db_files[train_split_index:val_split_index]
val_db_files = unique_db_files[val_split_index:]

def assign_split(db_file):
    if db_file in train_db_files:
        return 'few_shot_examples'
    elif db_file in val_db_files:
        return 'validation'
    else:
        return 'test'

# Assign 'split' column based on the db_file values
df['split'] = df['db_file'].apply(assign_split)

# Validate the split
train_count = df[df['split'] == 'few_shot_examples'].shape[0]
val_count = df[df['split'] == 'validation'].shape[0]
test_count = df[df['split'] == 'test'].shape[0]

print(f"Train set size: {train_count}")
print(f"Validation set size: {val_count}")
print(f"Test set size: {test_count}")
print(f"Unique db_files in train: {df[df['split'] == 'few_shot_examples']['db_file'].nunique()}")
print(f"Unique db_files in validation: {df[df['split'] == 'validation']['db_file'].nunique()}")
print(f"Unique db_files in test: {df[df['split'] == 'test']['db_file'].nunique()}")
print(f"Overlap in db_files (train & validation): {set(df[df['split'] == 'few_shot_examples']['db_file']).intersection(set(df[df['split'] == 'validation']['db_file']))}")
print(f"Overlap in db_files (train & test): {set(df[df['split'] == 'few_shot_examples']['db_file']).intersection(set(df[df['split'] == 'test']['db_file']))}")
print(f"Overlap in db_files (validation & test): {set(df[df['split'] == 'validation']['db_file']).intersection(set(df[df['split'] == 'test']['db_file']))}")

# Save the updated dataframe to CSV
df.to_csv("data/ambrosia/data/ambrosia_resplit.csv", index=False)

# Count ambiguous vs unambiguous examples in each split
print("\nAmbiguity Counts (Train):")
print(df[df['split'] == 'few_shot_examples']['is_ambiguous'].value_counts())
print("\nAmbiguity Counts (Validation):")
print(df[df['split'] == 'validation']['is_ambiguous'].value_counts())
print("\nAmbiguity Counts (Test):")
print(df[df['split'] == 'test']['is_ambiguous'].value_counts())

# Count types of ambiguity in each split
print("\nAmbiguity Types Distribution:")

# Scope
print("Scope Train:", len(df[(df['split'] == 'few_shot_examples') & (df['ambig_type'] == 'scope')]))
print("Scope Validation:", len(df[(df['split'] == 'validation') & (df['ambig_type'] == 'scope')]))
print("Scope Test:", len(df[(df['split'] == 'test') & (df['ambig_type'] == 'scope')]))

# Attachment
print("Attachment Train:", len(df[(df['split'] == 'few_shot_examples') & (df['ambig_type'] == 'attachment')]))
print("Attachment Validation:", len(df[(df['split'] == 'validation') & (df['ambig_type'] == 'attachment')]))
print("Attachment Test:", len(df[(df['split'] == 'test') & (df['ambig_type'] == 'attachment')]))

# Vague
print("Vague Train:", len(df[(df['split'] == 'few_shot_examples') & (df['ambig_type'] == 'vague')]))
print("Vague Validation:", len(df[(df['split'] == 'validation') & (df['ambig_type'] == 'vague')]))
print("Vague Test:", len(df[(df['split'] == 'test') & (df['ambig_type'] == 'vague')]))