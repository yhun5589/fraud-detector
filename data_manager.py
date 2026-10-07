import pandas as pd

# Load everything
df_full = pd.read_csv("AIML Dataset.csv")        # your full dataset
df_train = pd.read_csv("train_300k.csv")     # original train
df_test = pd.read_csv("test_200k.csv")       # original test

print("Full size:", len(df_full))

# Combine used rows
df_used = pd.concat([df_train, df_test])

print("Already used:", len(df_used))

# Remove used rows from full dataset
# This method works even if order was random
df_remaining = df_full.merge(
    df_used.drop_duplicates(),
    how="outer",
    indicator=True
).query('_merge == "left_only"').drop(columns=['_merge'])

print("Remaining rows:", len(df_remaining))

# Carve 800k from remaining
df_800k = df_remaining.sample(n=800000, random_state=42)

# Split into 4 files × 200k
for i in range(4):
    start = i * 200000
    end = (i + 1) * 200000
    df_800k.iloc[start:end].to_csv(f"extra_{i+1}_200k.csv", index=False)

print("Done. 4 files created. Zero overlap.")