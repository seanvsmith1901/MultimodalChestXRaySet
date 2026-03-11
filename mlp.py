# the goal of this file is to train and pickle our MLP so that we can access it later.
import torch, torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

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
            nn.Sigmoid()  # need to squash to 0 or 1.
        )

    # using nn.sequential makes this look much neater.
    def forward(self, x):
        return self.network(x)        # gree hee hee.


# using HE initilization for now.
def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_normal_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)

# ripped directly from lab 5.
def train_mlp(model, data_loader, n_epoch=50):
    print("Training MLP")
    # Initialize model weights
    model.apply(weights_init)

    # store the loss and the % correct at each epoch
    losses_train = np.zeros((n_epoch))
    errors_train = np.zeros((n_epoch))
    losses_test = np.zeros((n_epoch))
    errors_test = np.zeros((n_epoch))

    # choose cross entropy loss function (equation 5.24)
    loss_function = nn.BCELoss()  # we want a litteld ifferent
    # construct SGD optimizer and initialize learning rate and momentum
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9)
    # object that decreases learning rate by half every 10 epochs
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)


    for epoch in range(n_epoch):
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
      predicted_class = (all_preds >= 0.5).float() # threshold at 0.5
      accuracy = 100 * (predicted_class == all_labels).float().mean()

      losses_train[epoch] = avg_loss
      errors_train[epoch] = 100 - accuracy

      # Run whole dataset to get statistics -- normally wouldn't do this
      # So we aren't! Simple as, really.
      # pred_train = model(x_train)
      # pred_test = model(x_test)
      # _, predicted_train_class = torch.max(pred_train.data, 1)
      # _, predicted_test_class = torch.max(pred_test.data, 1)
      # errors_train[epoch] = 100 - 100 * (predicted_train_class == y_train).float().sum() / len(y_train)
      # errors_test[epoch]= 100 - 100 * (predicted_test_class == y_test).float().sum() / len(y_test)
      # losses_train[epoch] = loss_function(pred_train, y_train).item()
      # losses_test[epoch]= loss_function(pred_test, y_test).item()
      # print(f'Epoch {epoch:5d}, train loss {losses_train[epoch]:.6f}, train error {errors_train[epoch]:3.2f},  test loss {losses_test[epoch]:.6f}, test error {errors_test[epoch]:3.2f}')

      # tell scheduler to consider updating learning rate
      model.train() # flip this back for the next epoch
      scheduler.step()

      print(f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | Train Acc: {accuracy:.1f}%")

    return errors_train, errors_test, losses_train, losses_test


def evaluate_mlp(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(x_test).squeeze(1) # our raw [0,1] probs
        predicted_class = (preds >= 0.5).float()

    acc = 100 * (predicted_class == y_test).float().mean().item()
    auc = roc_auc_score(y_test.numpy(), preds.numpy())
    print(f"Test Accuracy : {acc:.1f}%")
    print(f"Test AUC-ROC  : {auc:.4f}")
