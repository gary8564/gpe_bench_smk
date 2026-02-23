import argparse
import os
import json
import torch
import torch.nn as nn
import numpy as np
import h5py
import time
from torch.utils.data import TensorDataset, DataLoader
from psimpy.emulator import PPGaSP
from high_dim_gp.dr.vae import VAE
from high_dim_gp.utils.util_funcs import uncertainty_propagation
from high_dim_gp.utils.error_metrics import ErrorMetrics
from high_dim_gp.utils.plot import plot_prediction, plot_residuals, plot_error_maps

def loss_function(x, x_hat, mean, log_var, epoch=None, total_epochs=None,
                  beta=1.0, capacity_max=25.0, capacity_warmup=0.3, free_bits=None):
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='none')  # [B, D]
    recon_loss = recon_loss.sum(dim=1).mean()

    # KL per-dimension: shape [B, Z]
    kl_per_dim = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())

    # Optional free-bits: clamp per-dim contributions
    if free_bits is not None:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    # Sum over latent dims, then mean over batch
    kl_loss = kl_per_dim.sum(dim=1).mean()

    # Capacity schedule
    if epoch is not None and total_epochs is not None:
        warmup_T = int(capacity_warmup * total_epochs)
        if epoch < warmup_T:
            C = capacity_max * (epoch / max(1, warmup_T))
        else:
            C = capacity_max
    else:
        C = capacity_max

    # β-VAE with capacity
    loss = recon_loss + beta * torch.abs(kl_loss - C)
    return loss, recon_loss, kl_loss, C

def train_vae(model: VAE, Y_train: torch.Tensor, batch_size: int = 32, device: str = "cpu"):
    # Create DataLoader
    Y_train = Y_train.to(device)
    train_dataset = TensorDataset(Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    # Loss function and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-04, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-5
        )
    # Training
    losses = []
    recon_losses = []
    kl_losses = []
    num_epochs = 1000
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            outputs, mean, logvar = model(data)
            loss, recon_loss, kl_loss, C = loss_function(
                outputs, data, mean, logvar, 
                epoch=epoch, total_epochs=num_epochs,
                beta=0.1,              
                capacity_max=25.0,     
                capacity_warmup=0.3,   # 30% warmup
                free_bits=0.02         
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_recon = epoch_recon / len(train_loader)
        avg_kl = epoch_kl / len(train_loader)
        
        scheduler.step(avg_loss)
        
        losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                f"Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.2f}, C: {C:.2f}")
            
    # Encoding the data using the trained variational autoencoder
    model.eval()  
    with torch.no_grad(): 
        # Encode data
        Y_train_reduced = model.encode(Y_train.to(device))[0].cpu().detach().numpy()
    dr_processing_time = time.time() - start_time
    return Y_train_reduced, dr_processing_time

def main():
    # 1. Parse arguments
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--latent-dim", type=int, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--qoi", required=True, choices=["hmax", "vmax", "cmax"])
    args = p.parse_args()
    
    # 2. Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Load data from HDF5
    hdf5_file = os.path.join(args.input_dir, "data.h5")
    with h5py.File(hdf5_file, 'r') as f:
        X_train = f['train_X'][:]
        Y_train = f['train_Y'][:]
        X_test = f['test_X'][:]
        Y_test = f['test_Y'][:]
        # Load standardization parameters
        scaler_mean = f['output_scaler_mean'][:]
        scaler_scale = f['output_scaler_scale'][:]
        # Reconstruct image parameters from HDF5
        valid_cols = f["valid_indices"][:]
        rows = f.attrs.get("rows")
        cols = f.attrs.get("cols")
        hill_path = f.attrs.get("background_img_path") 
        if hill_path == "":
            hill_path = None
    output_img_params = {
                        "filtered_columns": valid_cols, 
                        "output_img_rows": rows, 
                        "output_img_cols": cols, 
                        "background_img_path": hill_path ,
                        }
        
    # 4. Train VAE
    Y_train_tensor = torch.FloatTensor(Y_train)
    input_dim = Y_train_tensor.shape[1]
    latent_dim = args.latent_dim
    model = model = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[2048, 1024, 256, 64] , device=args.device)
    Y_train_reduced, dr_processing_time = train_vae(model, Y_train_tensor, device=args.device)
    
    # 5. Train
    emulator = PPGaSP(ndim=X_train.shape[1])
    start_time = time.time()
    emulator.train(design=X_train, response=Y_train_reduced)
    gp_training_time = time.time() - start_time
    training_time = dr_processing_time + gp_training_time
    
    # 6. Predict
    start_time = time.time()
    predictions = emulator.predict(X_test)
    gp_infer_time = time.time() - start_time
    
    # 7. Reconstruct back to orginal space, descale and filter out predictive values below threshold
    start_time = time.time()
    predictions_mean, predictions_lower, predictions_upper, predictions_std = uncertainty_propagation(model, Y_train, predictions, scaler_mean, scaler_scale, device=args.device)
    postprocessing_time = time.time() - start_time
    infer_time = gp_infer_time + postprocessing_time
    ground_truth = Y_test * scaler_scale + scaler_mean
    ground_truth = np.where(ground_truth < args.threshold, 0, ground_truth)
    predictions_mean = np.where(predictions_mean < args.threshold, 0, predictions_mean)
    predictions_lower = np.where(predictions_mean < args.threshold, 0, predictions_lower)
    predictions_upper = np.where(predictions_mean < args.threshold, 0, predictions_upper)
    predictions_std = np.where(predictions_mean < args.threshold, 0, predictions_std)
    
    # 8. Compute evaluation metrics
    rmse = ErrorMetrics.RMSE(predictions=predictions_mean, observations=ground_truth)
    coverage_prob = ErrorMetrics.CoverageProbability(predictions_mean, predictions_lower, predictions_upper, ground_truth)
    quantile_coverage_error = ErrorMetrics.QuantileCoverageError(predictions_lower, predictions_upper, ground_truth)
    
    # 9. Visualize predictions and ground-truths
    model_name = "VAE-PPGaSP"
    predictions = np.dstack((predictions_mean, predictions_std))
    plot_prediction(ground_truth, predictions, model_name, output_dir) 
    plot_residuals(ground_truth, predictions, model_name, output_dir)
    plot_error_maps(ground_truth, predictions_mean, output_img_params, model_name, output_dir, args.qoi)

    # 10. Save metrics
    metrics = dict(
        name=model_name,
        rmse=float(rmse),
        coverage_prob=float(coverage_prob),
        quantile_coverage_error=float(quantile_coverage_error),
        train_time=float(training_time),
        infer_time=float(infer_time)
    )
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[VAE-PPGaSP] metrics → {os.path.join(output_dir, 'metrics.json')}")   
    
    # 11. Save predictions and ground-truth for later benchmarking analysis
    hdf5_file = os.path.join(output_dir, "pred_and_gt.h5")
    with h5py.File(hdf5_file, 'w') as f:
        f.create_dataset('pred_mean', data=predictions_mean.astype(np.float64))
        f.create_dataset('gt', data=ground_truth.astype(np.float64))
    print(f"[VAE-PPGaSP] wrote predictions and ground-truth → {hdf5_file}")
if __name__=="__main__":
    main()