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

    # using nn.sequential makes this look much neater.
    def forward(self, x):
        return self.network(x)        # gree hee hee.