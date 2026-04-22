# All of the MLP stuf we could possible need. 
import torch, torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import fbeta_score
import spacy # small model, don't throw it on GPU.
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from src.data.nlp_preprocessing import *
from src.training.train_mlp import train_mlp
from scripts.ensemble_evalute import evaluate_mlp
from src.data.preprocessing import build_nlp_pipeline
probability_flag = 0.50


class MLP(nn.Module):
    def __init__(self, input_nodes=500): # this is specified by the number of keywords that we need.
        super(MLP, self).__init__()

        self.network = nn.Sequential(
            # block 1
            nn.Linear(input_nodes, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # another one
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # convert it to output.
            nn.Linear(128, 1),
            # nn.Sigmoid()  # don't use this if you switch to BCEwithLogits, sigmoid already there.
        )
        self.probability_flag = 0.5

    # using nn.sequential makes this look much neater.
    def forward(self, x):
        return self.network(x)        # gree hee hee.


# using HE initilization for now.
def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_normal_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)

def create_x_stuff(df, mode):
    nlp = build_nlp_pipeline()
    num_epochs = 200

    # creates X and Y based on penumonia values and the DF.
    # print("starting the model creation process ")
    # this returns X and y as df's to try and use them in the train test split and dataloaders. Don't think its working as intended.

    # this uses a tokenizer to create a better representation of X and Y.

    X, y, df = get_clean_stuff(df, nlp, mode)  # returns a clean X and y that we will need for training later.

    # print("Here is our shape ", X.shape)
    # print("Post conversion")

    # now I gotta create data loaders. Bleh.
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    # NOW we apply smote to prevent data leakage.
    # gonna try adding smote.
    smote = SMOTE(random_state=42)  # trying to fix some class imbalance

    X_train_bal, y_train_bal = smote.fit_resample(X_train_raw, y_train_raw)

    # print(f"Before SMOTE: {X.shape[0]} samples, {y.sum()} positive")
    # print(f"After SMOTE:  {X_train_bal.shape[0]} samples, {y_train_bal.sum()} positive")

    # we GOTTA turn these back into tensors before torch notices!
    X_train = torch.tensor(X_train_bal, dtype=torch.float32)
    y_train = torch.tensor(y_train_bal, dtype=torch.float32)
    X_test = torch.tensor(X_test_raw, dtype=torch.float32)
    y_test = torch.tensor(y_test_raw, dtype=torch.float32)

    return X_train, y_train, X_test, y_test, X, df


def create_mlp_model_from_splits(train_df, val_df, test_df, mode, text_amount="medium", model_type="singe"):

    nlp = build_nlp_pipeline()
    num_epochs = 50


    train_df = prepare_text_features(train_df.copy(), nlp, mode=text_amount)
    val_df = prepare_text_features(val_df.copy(), nlp, mode=text_amount)
    test_df = prepare_text_features(test_df.copy(), nlp, mode=text_amount)

    X_train, y_train, X_val, y_val, X_test, y_test, vectorizer = create_tfidf_features(
        train_df, val_df=val_df, test_df=test_df, mode=text_amount
    )

    # just use smote, can't hurt can it?
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    import pickle
    vectorizer_path = f"../models/vectorizer_{mode}_{text_amount}.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"✅ Vectorizer saved to: {vectorizer_path}")

    input_nodes = X_train.shape[1]  # ← This is where X is defined!
    model = MLP(input_nodes)

    return create_mlp(X_train, y_train, X_val, y_val, X_test, y_test, num_epochs, model)



def create_mlp_model(df, mode, model_type):
    X_train, y_train, X_test, y_test, X, df = create_x_stuff(df, mode)
    num_epochs = 200
    input_nodes = X.shape[1]
    model = MLP(input_nodes) # Do this here. Trust me. 

    if model_type == "single":
        model, errors_train, errors_test, losses_train, losses_test = create_mlp(X, X_train, y_train, X_test, y_test,
                                                                                 num_epochs, model)

    # gonna be honest... you're probably not super worth running.
    elif model_type == "ensemble":
        model, errors_train, errors_test, losses_train, losses_test = create_mlp_ensemble(X, X_train, y_train, X_test, y_test,
                                                                                 num_epochs, model)
    else:
        print("something not right")
        model, errors_train, errors_test, losses_train, losses_test = create_mlp(X, X_train, y_train, X_test, y_test,
                                                                                 num_epochs, model)


    # lets try to get a rough idea of where we are.


    return model, errors_train, errors_test, losses_train, losses_test  # get everything back.

# no longer accepts X as an argument. 
def create_mlp(X_train, y_train, X_val, y_val, X_test, y_test, num_epochs, model):

    #  cast htem into ints
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)


    data_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100, shuffle=True)

    errors_train, errors_test, losses_train, losses_test = train_mlp(model, data_loader, y_train, num_epochs)

    # evaluate_mlp(model, X_test, y_test)

    return model.state_dict(), errors_train, errors_test, losses_train, losses_test  # get everything back.


def create_mlp_ensemble(X, X_train, y_train, X_test, y_test, num_epochs):
    all_probs = []
    num_models = 5

    for i in range(num_models):
        print("training model ", i)

        model = MLP(input_nodes=X_train.shape[1])
        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100, shuffle=True,)

        train_mlp(model, loader, y_train, num_epochs)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_test).squeeze(1))
            all_probs.append(probs.numpy())

    ensemble_probs = np.mean(all_probs, axis=0)
    return ensemble_probs, None, None, None, None



