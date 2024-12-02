import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataloader import load_data
from src.model import PredatorPreyODE
from src.training import training_loop, evaluate_model
torch.manual_seed(0)


def main():

    data_tensor = load_data('/Users/subedi/Documents/ie-node/data/data_hare_lynx.csv')
    num_time_steps = data_tensor.size()[0]
    data_tensor = data_tensor.float()

    predator_prey_ode = PredatorPreyODE()
    initial_population = data_tensor[0]
    time_points = torch.linspace(0, num_time_steps, steps=num_time_steps)  # Placeholder time points for testing

    with torch.no_grad():
        output = odeint(predator_prey_ode, initial_population, time_points)
        print("Model output shape: ", output.shape)
    mse_loss = nn.MSELoss()
    optimier = optim.Adam(predator_prey_ode.parameters(), lr=0.01)
    num_epochs = 1000

    model = training_loop(predator_prey_ode, optimier, data_tensor, num_epochs, mse_loss)
    final_loss = evaluate_model(model, data_tensor, mse_loss)
    print(f"Final loss: {final_loss:.4f}")

if __name__ == "__main__":
    main()