import numpy as np
import torch
import pandas as pd
def load_data(file_path, normalise=True):
    data = np.array(pd.read_csv(file_path))
    data = data[:,1:3] # Drop time step : why?

    data_normalized = np.copy(data)
    for i in range(data.shape[1]):
        data_log = np.log(data[:,i])
        data_normalized[:,i] = (data_log - data_log.mean()) / data_log.std()
    
    ## Convert data to PyTorch tensor
    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
    
    return data_tensor
