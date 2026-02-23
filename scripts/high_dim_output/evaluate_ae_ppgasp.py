import argparse
import os
import json
import torch
import torch.nn as nn
import numpy as np
import h5py
import time
from psimpy.emulator import PPGaSP
from high_dim_gp.dr.ae import AutoEncoder
from high_dim_gp.utils.util_funcs import uncertainty_propagation
from high_dim_gp.utils.error_metrics import ErrorMetrics
from high_dim_gp.utils.plot import plot_prediction, plot_residuals, plot_error_maps

def train_ae(model: AutoEncoder, Y_train: torch.Tensor, device: str = "cpu"):
    Y_train = Y_train.to(device)
    model = model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # Training
    losses = []
    num_epochs = 500
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        outputs = model(Y_train)
        loss = loss_function(outputs, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    # Encoding the data using the trained variational autoencoder
    model.eval()
    with torch.no_grad():
        Y_train_reduced = model.encoder(Y_train).detach().cpu().numpy()
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
            "background_img_path": hill_path,
            }
        
    # 4. Train AE
    Y_train_tensor = torch.FloatTensor(Y_train)
    input_dim = Y_train_tensor.shape[1]
    latent_dim = args.latent_dim
    model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[1024, 256, 64, 64, 16])
    Y_train_reduced, dr_processing_time = train_ae(model, Y_train_tensor, device=args.device)
    
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
    model_name = "AE-PPGaSP"
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
    with open(os.path.join(output_dir,"metrics.json"),"w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[AE-PPGaSP] metrics → {output_dir}/metrics.json")
    
    # 11. Save predictions and ground-truth for later benchmarking analysis
    hdf5_file = os.path.join(output_dir, "pred_and_gt.h5")
    with h5py.File(hdf5_file, 'w') as f:
        f.create_dataset('pred_mean', data=predictions_mean.astype(np.float64))
        f.create_dataset('gt', data=ground_truth.astype(np.float64))
    print(f"[AE-PPGaSP] wrote predictions and ground-truth → {hdf5_file}")

if __name__=="__main__":
    main()