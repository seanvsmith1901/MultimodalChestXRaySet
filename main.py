import pandas as pd
from dataloaderCreator import *
from tokenizationProcess import * # need
import spacy # small model, don't throw it on GPU.
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

from mlp import MLP # bring our custom class in.

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

    # having these separate might actually be better.
    df = build_master_df(projections_path, reports_path)

    # ok now we have to do the freaking NLP stuff in an attempt to understand this better.
    nlp = spacy.load("en_core_web_sm")

    X, y = get_clean_stuff(df, nlp) # returns a clean X and y that we will need for training later.

    # X = pd.DataFrame(X)
    # y = pd.DataFrame(y)
    # print(X.sample(5))
    # print(y.sample(5))

    input_nodes = X.shape[1]

    model = MLP(input_nodes)

    # DO REMEMBER THAT THIS ONLY DOES PNEUMONIA ATM
    # train_loader, test_loader = get_dataloaders(projections_path, reports_path, image_dir)

    # so for our VERY VERY First run, we are going to
    # create an MLP that assings a triage score based off of written text
    # create our OWN CNN that tells us how likely they are to have some pneumonia
    # and then combine them for a triage score.
