import pandas as pd
from dataloaderCreator import *
from evaluationSuite import plot_training_curves
from tokenizationProcess import * # need
import spacy # small model, don't throw it on GPU.
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from mlp import create_mlp_model, create_x_stuff

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

    # print("Creating the MLP")
    # # ok now we have to do the freaking NLP stuff in an attempt to understand this better.
    # modes = ["sparse", "medium", "full"]
    modes = ["medium"]
    # little jury rigging here.
    # X_train, y_train, X_test, y_test, X, df =  create_x_stuff(old_df, modes[0]) # thats the fethcer.
    # X_train_np = X_train.numpy()
    # y_train_np = y_train.numpy()
    # X_test_np = X_test.numpy()
    # y_test_np = y_test.numpy()

    # xgb = XGBClassifier(n_estimators=500, scale_pos_weight=3, random_state=42)
    # xgb.fit(X_train_np, y_train_np)
    # print(f"XGB AUC: {roc_auc_score(y_test_np, xgb.predict_proba(X_test_np)[:,1]):.4f}")
    #
    # model_type = "ensemble"
    # ensemble_probs, _, _, _, _ = create_mlp_model(df, modes[0], model_type)
    # print(f"Ensemble AUC: {roc_auc_score(y_test_np, ensemble_probs):.4f}")
    #
    # model_type = "single"
    # model, errors_train, errors_test, losses_train, losses_test = create_mlp_model(df, modes[0], model_type)
    # # plot_training_curves(losses_train, errors_train)

    model_type = "single"
    for mode in modes:
        model, errors_train, errors_test, losses_train, losses_test = create_mlp_model(df, mode, model_type)
        plot_training_curves(losses_train, errors_train)


