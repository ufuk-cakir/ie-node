import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import seaborn as sns

def training_loop(model, optimizer, data, epochs, loss_fn):

    model.train()
    # model.eval()

    ## Get time steps of data
    time_points = torch.linspace(0, data.shape[0], steps=data.shape[0])  # Time points based on data length

    for epoch in range(epochs):

        ## Set gradients to zero before each pass
        optimizer.zero_grad()

        ## Forward pass: use ODE solver to predict populations over time
        predicted_population = odeint(model, data[0], time_points)

        ## Compute loss as MSE between predicted and actual data
        loss = loss_fn(predicted_population, data)

        ## Backward pass: compute gradients and update model parameters
        loss.backward()
        optimizer.step()

        ## Log training progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    return model
def evaluate_model(model, data, loss):

    ## Get time steps of data
    time_points = torch.linspace(0, data.shape[0], steps=data.shape[0])  # Time points based on data length
    
    ## Perform forward pass to get model predictions
    predicted_population = odeint(model, data[0], time_points)

    ## Calculate final MSE loss on the dataset
    final_loss = loss(predicted_population, data).item()
    print(f"Final Loss: {final_loss:.4f}")

    ## Visualize the results
    plt.figure(figsize=(14,6))
    plt.plot(data[:, 0], '--o', label="True Hare Population", color='royalblue')
    plt.plot(data[:, 1], '--o', label="True Lynx Population", color='salmon')
    plt.plot(predicted_population[:, 0].detach(), '-', label="Predicted Hare Population", color='royalblue')
    plt.plot(predicted_population[:, 1].detach(), '-', label="Predicted Lynx Population", color='salmon')
    plt.xlabel('Year')
    plt.ylabel('Normalized Population')
    plt.legend()
    plt.title('True vs Predicted Hare-Lynx Population Dynamics')
    sns.despine()
    plt.show()

    return final_loss
