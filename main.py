import pandas as pd
from dataloaderCreator import *
from tokenizationProcess import * # need
import spacy # small model, don't throw it on GPU.
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from createMLPStuff import create_mlp_model # we will want this.

from mlp import MLP, evaluate_mlp  # bring our custom class in.

# AIGHT SO WE DID ALL OF THIS WRONGF
# well kind of. we now have the idea of X and y and all that
# as well as a basic linear model and whatnot
# form here, we actually need to create the DF first, split into train test
# then create X from that.

if __name__ == "__main__":

    # this contains the metadata that we want for our MLP
    projections_path = 'IU-xray/indiana_projections.csv'
    # this contains the labelling we want for our image CNN
    reports_path = 'IU-xray/indiana_reports.csv'
    # and of course, the images
    image_dir = 'IU-xray/images/images_normalized'

    print("Creating the DF")
    df = build_master_df(projections_path, reports_path)
    print(df['pneumonia_label'].value_counts())
    print(df['pneumonia_label'].value_counts(normalize=True))

    print("Creating the MLP")
    # ok now we have to do the freaking NLP stuff in an attempt to understand this better.
    nlp = spacy.load("en_core_web_sm")
    num_epochs = 20
    mlp_model = create_mlp_model(df, nlp, num_epochs)

