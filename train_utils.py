import numpy as np
import torch

def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str = 'cpu'):
    """
    Returns a batch of data from the input array.
    
    Parameters:
    - dataset: Input numpy array.
    - batch_size: Number of samples in the batch.
    - context_length: Length of the context for each sample.
    - device: Device to which the batch should be moved (default is 'cpu').
    
    Returns:
    - A tuple containing the batch of data and its corresponding labels.
    """
    indices = np.random.choice(len(dataset) - context_length, size=batch_size, replace=False)
    batch_indices = indices[:, None] + np.arange(context_length)
    label_indices = indices[:, None] + np.arange(1, context_length + 1)
    batch = torch.from_numpy(dataset[batch_indices]).to(device)
    labels = torch.from_numpy(dataset[label_indices]).to(device)
    
    return batch, labels


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, out):
    """
    Saves the model and optimizer state to a checkpoint file.
    
    Parameters:
    - model: The model to save.
    - optimizer: The optimizer to save.
    - epoch: The current epoch number.
    - out: The output path for the checkpoint file.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Loads the model and optimizer state from a checkpoint file.
    
    Parameters:
    - model: The model to load the state into.
    - optimizer: The optimizer to load the state into.
    - in_path: The input path for the checkpoint file.
    
    Returns:
    - epoch: The epoch number from the checkpoint.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return epoch