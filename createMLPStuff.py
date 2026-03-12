# we actually need to keep these entirely separate
# like the CNN and MLP
# so for now we are workign our way through the MLP and going from there.
# so lets do that.
from tokenizationProcess import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from mlp import * # the actual class that does the magic and the training function.
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE

def create_mlp_model(df, nlp, num_epochs):
    # creates X and Y based on penumonia values and the DF.
    print("starting the model creation process ")
    # this returns X and y as df's to try and use them in the train test split and dataloaders. Don't think its working as intended.

    # this uses a tokenizer to create a better representation of X and Y.
    X, y = get_clean_stuff(df, nlp)  # returns a clean X and y that we will need for training later.





    print("Here is our shape ", X.shape)
    print("Post conversion")

    # now I gotta create data loaders. Bleh.
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    # NOW we apply smote to prevent data leakage.

    # gonna try adding smote.
    smote = SMOTE(random_state=42) # trying to fix some class imbalance

    X_train_bal, y_train_bal = smote.fit_resample(X_train_raw, y_train_raw)

    print(f"Before SMOTE: {X.shape[0]} samples, {y.sum()} positive")
    print(f"After SMOTE:  {X_train_bal.shape[0]} samples, {y_train_bal.sum()} positive")

    # we GOTTA turn these back into tensors before torch notices!
    X_train = torch.tensor(X_train_bal, dtype=torch.float32)
    y_train = torch.tensor(y_train_bal, dtype=torch.float32)
    X_test = torch.tensor(X_test_raw, dtype=torch.float32)
    y_test = torch.tensor(y_test_raw, dtype=torch.float32)

    print("Post split")

    print(f"Train positives: {y_train.sum().item()}")
    print(f"Train negatives: {(y_train == 0).sum().item()}")

    print(f"Test positives: {y_test.sum().item()}")
    print(f"Test negatives: {(y_test == 0).sum().item()}")

    input_nodes = X.shape[1]
    model = MLP(input_nodes)
    data_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100, shuffle=True, worker_init_fn=np.random.seed(1))

    errors_train, errors_test, losses_train, losses_test = train_mlp(model, data_loader, y_train, num_epochs)



    # lets try to get a rough idea of where we are.
    evaluate_mlp(model, X_test, y_test)

    return model, errors_train, errors_test, losses_train, losses_test # get everything back.