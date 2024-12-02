from torch import nn

class PredatorPreyODE(nn.Module):

    def __init__(self):
        super(PredatorPreyODE, self).__init__()

        ## Define a small feed-forward neural network with one hidden layer
        self.net = nn.Sequential(
            nn.Linear(2, 50),   # Input layer with two variables: hare and lynx populations
            nn.Tanh(),
            nn.Linear(50, 2)    # Output layer with two outputs: dHare/dt and dLynx/dt
        )

    def forward(self, t, x):
        return self.net(x)
