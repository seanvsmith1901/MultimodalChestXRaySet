import pandas as pd
from dataloaderCreator import *
from tokenizationProcess import * # need
import spacy # small model, don't throw it on GPU.
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


if __name__ == "__main__":

    # this contains the metadata that we want for our MLP
    projections_path = 'IU-xray/indiana_projections.csv'
    # this contains the labelling we want for our image CNN
    reports_path = 'IU-xray/indiana_reports.csv'
    # and of course, the images
    image_dir = 'IU-xray/images/images_normalized'

    # having these separate might actually be better.
    df = build_master_df(projections_path, reports_path)
    print("IF this doesn't return true kill me ", spacy.prefer_gpu()) # plz work.

    # ok now we have to do the freaking NLP stuff in an attempt to understand this better.
    nlp = spacy.load("en_core_web_sm")

    X, y = get_clean_stuff(df, nlp) # returns a clean X and y that we will need for training later.

    pickle.dump(X, open("X.pkl", "wb"))
    pickle.dump(y, open("y.pkl", "wb"))

