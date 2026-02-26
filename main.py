import pandas as pd

df_train = pd.read_csv("data/train_fixed.csv")
df_validation = pd.read_csv("data/validation_fixed.csv")

print("here is the len of df train ", len(df_train))
print("here are the cols of df train ", df_train.columns)