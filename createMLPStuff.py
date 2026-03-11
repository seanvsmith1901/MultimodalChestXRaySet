# we actually need to keep these entirely separate
# like the CNN and MLP
# so for now we are workign our way through the MLP and going from there.
# so lets do that.
from tokenizationProcess import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from mlp import * # the actual class that does the magic and the training function.
from torch.utils.data import TensorDataset, DataLoader

def create_mlp_model(df, nlp, num_epochs):
    # creates X and Y based on penumonia values and the DF.
    print("starting the model creation process ")
    # this returns X and y as df's to try and use them in the train test split and dataloaders. Don't think its working as intended.
    X, y = get_clean_stuff(df, nlp)  # returns a clean X and y that we will need for training later.

    print("Here is our shape ", X.shape)

    X = torch.tensor(X, dtype=torch.float32)  # prefer tensor() over Tensor()
    y = torch.tensor(y, dtype=torch.float32)
    print("Post conversion")

    # now I gotta create data loaders. Bleh.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print("Post split")

    input_nodes = X.shape[1]
    model = MLP(input_nodes)

    data_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100, shuffle=True, worker_init_fn=np.random.seed(1))

    errors_train, errors_test, losses_train, losses_test = train_mlp(model, data_loader, num_epochs)

    # lets try to get a rough idea of where we are.
    evaluate_mlp(model, X_test, y_test)

    return model, errors_train, errors_test, losses_train, losses_test # get everything back.