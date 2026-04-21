# the goal of this file is to train and pickle our MLP so that we can access it later.
import torch, torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import fbeta_score
import spacy # small model, don't throw it on GPU.
from legacy.Scripts.tokenizationProcess import get_clean_stuff, build_nlp_pipeline
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader




class MLP(nn.Module):
    def __init__(self, input_nodes=500, probability_flag=0.5): # this is specified by the number of keywords that we need.
        super(MLP, self).__init__()
        self.probability_flag = probability_flag

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
    print("starting the model creation process ")
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


def create_mlp_model(df, mode, model_type):
    X_train, y_train, X_test, y_test, X, df = create_x_stuff(df, mode)
    num_epochs = 200

    # print("Post split")
    #
    # print(f"Train positives: {y_train.sum().item()}")
    # print(f"Train negatives: {(y_train == 0).sum().item()}")
    #
    # print(f"Test positives: {y_test.sum().item()}")
    # print(f"Test negatives: {(y_test == 0).sum().item()}")

    if model_type == "single":
        model, errors_train, errors_test, losses_train, losses_test = create_mlp(X, X_train, y_train, X_test, y_test,
                                                                                 num_epochs)

    # gonna be honest... you're probably not super worth running.
    # elif model_type == "ensemble":
    #     model, errors_train, errors_test, losses_train, losses_test = create_mlp_ensemble(X, X_train, y_train, X_test, y_test,
    #                                                                              num_epochs)
    else:
        print("something not right")
        model, errors_train, errors_test, losses_train, losses_test = create_mlp(X, X_train, y_train, X_test, y_test,
                                                                                 num_epochs)


    # lets try to get a rough idea of where we are.


    return model, errors_train, errors_test, losses_train, losses_test  # get everything back.

def create_mlp(X, X_train, y_train, X_test, y_test, num_epochs):

    input_nodes = X.shape[1]
    model = MLP(input_nodes)
    data_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100, shuffle=True,
                             worker_init_fn=np.random.seed(1))

    errors_train, errors_test, losses_train, losses_test = train_mlp(model, data_loader, y_train, num_epochs)

    # lets try to get a rough idea of where we are.
    evaluate_mlp(model, X_test, y_test)

    return model, errors_train, errors_test, losses_train, losses_test  # get everything back.


# def create_mlp_ensemble(X, X_train, y_train, X_test, y_test, num_epochs):
#     all_probs = []
#     num_models = 5
#
#     for i in range(num_models):
#         print("training model ", i)
#
#         model = MLP(input_nodes=X_train.shape[1])
#         loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100, shuffle=True,)
#
#         train_mlp(model, loader, y_train, num_epochs)
#         model.eval()
#         with torch.no_grad():
#             probs = torch.sigmoid(model(X_test).squeeze(1))
#             all_probs.append(probs.numpy())
#
#     ensemble_probs = np.mean(all_probs, axis=0)
#     return ensemble_probs, None, None, None, None





# gonna try to add some weird anti overfitting check to it, watch out.
def train_mlp(model, data_loader, y_train, n_epoch=50, patience=5):

    print("Training MLP")
    # Initialize model weights
    model.apply(weights_init)

    # store the loss and the % correct at each epoch
    losses_train = np.zeros((n_epoch))
    errors_train = np.zeros((n_epoch))


    # choose cross entropy loss function (equation 5.24)
    # TRYING SOMETHING NEW SIRE
    n_neg = (y_train == 0).sum().item()
    n_pos = (y_train == 1).sum().item()
    pos_weight = torch.tensor(n_neg / n_pos)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # trying a different loss function, we shall see.
    # construct SGD optimizer and initialize learning rate and momentum
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9)
    # object that decreases learning rate by half every 10 epochs
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # early stopping stuff coming in hot.
    best_loss = np.inf
    best_weights = None
    patience_counter = 0
    epochs_completed = 0

    for epoch in range(n_epoch):
      epochs_completed += 1
      model.train()
      epoch_loss = 0
      # loop over batches
      for i, batch in enumerate(data_loader):
        # retrieve inputs and labels for this batch
        x_batch, y_batch = batch
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass -- calculate model output
        pred = model(x_batch).squeeze(1) # this is supposed to help
        # compute the loss
        loss = loss_function(pred, y_batch)
        # backward pass
        loss.backward()
        # SGD update
        optimizer.step()
        epoch_loss += loss.item()


      model.eval()
      with torch.no_grad():
          all_preds, all_labels = [], []
          for x_batch, y_batch in data_loader:
              pred = model(x_batch).squeeze(1)
              all_preds.append(pred)
              all_labels.append(y_batch)

      all_preds = torch.cat(all_preds)
      all_labels = torch.cat(all_labels)

      avg_loss = epoch_loss / len(data_loader)

      all_probs = torch.sigmoid(all_preds) # convert back to probabilities from logits in the loss function .

      predicted_class = (all_probs >= model.probability_flag).float() # threshold at 0.5
      accuracy = 100 * (predicted_class == all_labels).float().mean()

      losses_train[epoch] = avg_loss
      errors_train[epoch] = 100 - accuracy

      # early stopping check
      if avg_loss < best_loss - 1e-4: # as a place to start
          best_loss = avg_loss
          best_weights = model.state_dict().copy() # save our best weights as a checkpoint
          patience_counter = 0
      else:
          patience_counter += 1
          if patience_counter >= patience:
              # print(f"Early stopping requried sire. At epoch {epoch+1} — best loss: {best_loss:.4f}")
              model.load_state_dict(best_weights) # restore our previous best model
              # trim the zero padded tail off the arrays
              losses_train = losses_train[:epoch + 1]
              errors_train = errors_train[:epoch + 1]
              break


      model.train() # flip this back for the next epoch
      # tell scheduler to consider updating learning rate
      scheduler.step()

      # print(f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | Train Acc: {accuracy:.1f}%")

    return (errors_train[:epochs_completed],
            errors_train[:epochs_completed],
            losses_train[:epochs_completed],
            losses_train[:epochs_completed])


def evaluate_mlp(model, x_test, y_test):
    print("************************************")
    model.eval()
    with torch.no_grad():
        logits = model(x_test).squeeze(1)
        preds  = torch.sigmoid(logits)
        predicted_class = (preds >= model.probability_flag).float()

    acc = 100 * (predicted_class == y_test).float().mean().item()
    auc = roc_auc_score(y_test.numpy(), preds.numpy())

    # this is what actually matters
    cm = confusion_matrix(y_test.numpy(), predicted_class.numpy())
    tn, fp, fn, tp = cm.ravel()

    for threshold in [0.5, 0.3, 0.2, 0.1, 0.05]:
        predicted_class = (preds >= threshold).float()
        cm = confusion_matrix(y_test.numpy(), predicted_class.numpy())
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(
            f"  t={threshold:.2f} | Sensitivity: {sensitivity:.1%} | Specificity: {specificity:.1%} | FN: {fn} | FP: {fp}")

    print(f"Test Accuracy : {acc:.1f}%")
    print(f"Test AUC-ROC  : {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (correctly said normal)   : {tn}")
    print(f"  False Positives (incorrectly flagged)      : {fp}")
    print(f"  False Negatives (missed abnormal cases)   : {fn}  ← this is the dangerous one")
    print(f"  True Positives  (correctly caught)         : {tp}")
    print(f"\nSensitivity (recall on positives): {tp/(tp+fn):.1%}")
    print(f"Specificity (recall on negatives): {tn/(tn+fp):.1%}")

    # say that missing a sick paitent is twice as bad as a false alarm
    # we can use this to try and skew things certian ways. Right now beta is set to 2.
    # beta = 2
    # for threshold in [0.50, 0.30, 0.20, 0.10, 0.05]:
    #     predicted_class = (preds >= threshold).float().numpy()
    #     f2 = fbeta_score(y_test.numpy(), predicted_class, beta=beta)
    #     print(f"t={threshold:.2f} | F2: {f2:.4f}")



