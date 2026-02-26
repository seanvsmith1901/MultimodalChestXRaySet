import pandas as pd
import glob


df_train = pd.read_csv("data/train_fixed.csv")
# print("this is the shape of train", df_train.shape)
#
df_validation = pd.read_csv("data/validation_fixed.csv")
# print("this is the shape of validation", df_validation.shape)
#
# print("this is the head of train ", df_train.head())
# print("this is the head of validation", df_validation.head())

# first_column_train = df_train.iloc[:,0] # grab the first column all the way down
# new_names = []
# for column in first_column_train:
#     filepath = column.split("/") # grab the first instance here
#     filepath.pop(0)
#     string_path = "/".join(filepath)
#     new_names.append("data/" + string_path)

df_train['Path'] = df_train['Path'].str.replace(
    "CheXpert-v1.0-small/", "data/", regex=False
)

df_validation['Path'] = df_validation['Path'].str.replace(
    "CheXpert-v1.0-small/", "data/", regex=False
)

df_train.to_csv("data/train_fixed.csv", index=False)
df_validation.to_csv("data/validation_fixed.csv", index=False)