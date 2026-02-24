import pandas as pd
import glob


df_train = pd.read_csv("data/train.csv")
print("this is the shape of train", df_train.shape)

df_validation = pd.read_csv("data/valid.csv")
print("this is the shape of validation", df_validation.shape)

print("this is the head of train ", df_train.head())
print("this is the head of validation", df_validation.head())

filesmatchingPattern = glob.glob("data/*.csv")
print("here is the len of filesmatchingpattern ", len(filesmatchingPattern))