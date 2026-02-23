import numpy as np
from high_dim_gp.dr.pca import InputDimReducer
import copy
import torch
import torch.nn as nn

def reduced_dim(reducer: InputDimReducer | None, X: np.ndarray) -> int:
    """Helper function to obtain the input dimension

    Args:
        reducer (InputDimReducer | None): Dimensionality reduction
        X (np.ndarray): Input dataset

    Returns:
        int: number of input dimensions
    """
    if reducer is None:
        return X.shape[1]
    r = copy.deepcopy(reducer)
    r.reducer.fit(X, show_cum_var_plot=False)
    n_comp = r.reducer.n_components
    return int(n_comp) 
    
def uncertainty_propagation(model: nn.Module, 
                            train_Y: np.ndarray, 
                            predictions: np.ndarray, 
                            scaler_mean: np.ndarray, 
                            scaler_std: np.ndarray,
                            num_mc_samples: int = 100, 
                            device: str = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to propagate the uncertainty from the latent space to the original space for AE/VAE.

    Args:
        model: AE/VAE
        train_Y: Training output data
        predictions: Predictions in the latent space
        scaler_mean: Mean of the output scaler
        scaler_std: Standard deviation of the output scaler
        num_mc_samples: Number of Monte Carlo samples
        device: computing device
        
    Returns:
        predictions_mean_original: Predictions mean in original space
        lower_CI: Lower confidence interval
        upper_CI: Upper confidence interval
        std_original: Standard deviation of the original space
    """
    model = model.to(device)
    original_dim = train_Y.shape[1]
    mean_reduced = predictions[:, :, 0]
    std_reduced = predictions[:, :, 3]
    num_test, latent_dim = mean_reduced.shape
    # Decode predictions
    if hasattr(model, "decoder"):
        predictions_mean_normalized = model.decoder(torch.FloatTensor(predictions[:, :, 0]).to(device)).detach().cpu().numpy()
    elif hasattr(model, "decode"):
        predictions_mean_normalized = model.decode(torch.FloatTensor(predictions[:, :, 0]).to(device)).cpu().detach().numpy()
    else:
        raise ValueError("Model does not have a decoder or decode method")
    predictions_mean_original = predictions_mean_normalized * (scaler_std + 1e-8) + scaler_mean
    # Generate samples in reduced space
    rng = np.random.default_rng()
    latent_samples = rng.normal(loc=mean_reduced[:, :, None], scale=std_reduced[:, :, None], size=(num_test, latent_dim, num_mc_samples))
    # Reconstruct original output space
    original_samples = np.zeros((num_test, original_dim, num_mc_samples))
    latent_flat = latent_samples.transpose(0, 2, 1).reshape(num_test * num_mc_samples, latent_dim)
    if hasattr(model, "decoder"):
        original_flat = model.decoder(torch.FloatTensor(latent_flat).to(device)).detach().cpu().numpy()
    elif hasattr(model, "decode"):
        original_flat = model.decode(torch.FloatTensor(latent_flat).to(device)).cpu().detach().numpy()
    else:
        raise ValueError("Model does not have a decoder or decode method")
    original_flat = original_flat * (scaler_std + 1e-8) + scaler_mean
    original_samples = original_flat.reshape(num_test, num_mc_samples, original_dim).transpose(0, 2, 1)
    # Compute statistics across samples
    std_original = np.std(original_samples, axis=2)
    margin = 1.96 * std_original
    lower_CI = predictions_mean_original - margin
    upper_CI = predictions_mean_original + margin 
    return predictions_mean_original, lower_CI, upper_CI, std_original