import argparse
import os
import json
import torch
import numpy as np
import h5py
from high_dim_gp.emulator.dkl import ExactGP
from high_dim_gp.utils.error_metrics import ErrorMetrics
from high_dim_gp.utils.plot import plot_prediction

def descale_data(gt_scaled: np.ndarray,
                 mean_scaled: np.ndarray,
                 std_scaled: np.ndarray,
                 lower95_scaled: np.ndarray,
                 upper95_scaled: np.ndarray,
                 scaler_mean: np.ndarray,
                 scaler_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Descale the data to the original scale
    
    Args:
        gt_scaled: The scaled test data
        mean_scaled: The mean of the scaled test data
        std_scaled: The standard deviation of the scaled test data
        lower95_scaled: The lower 95% confidence interval of the scaled test data
        upper95_scaled: The upper 95% confidence interval of the scaled test data
        scaler_mean: Mean values from standardization
        scaler_scale: Scale values from standardization
        
    Returns:
        groud_truth: The original test data
        mean: The mean of the original test data
        std: The standard deviation of the original test data
        lower95: The lower 95% confidence interval of the original test data
        upper95: The upper 95% confidence interval of the original test data
    """
    mu = scaler_mean[-1] 
    sigma = scaler_scale[-1] 
    groud_truth = gt_scaled * sigma + mu
    mean = mean_scaled * sigma + mu
    std = std_scaled * sigma 
    lower95 = lower95_scaled * sigma + mu
    upper95 = upper95_scaled * sigma + mu
    return groud_truth, mean, std, lower95, upper95

def main():
    # 1. Parse arguments
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    
    # 2. Load data from HDF5
    hdf5_file = os.path.join(args.input_dir, "data.h5")
    with h5py.File(hdf5_file, 'r') as f:
        X_train = torch.from_numpy(f['train_X'][:]).float()
        Y_train = torch.from_numpy(f['train_Y'][:]).float()
        X_test = torch.from_numpy(f['test_X'][:]).float()
        Y_test = torch.from_numpy(f['test_Y'][:]).float()
        
        # Load standardization parameters
        scaler_mean = f['scaler_mean'][:]
        scaler_scale = f['scaler_scale'][:]
    
    # 3. Train
    model = ExactGP(device=args.device,
                    kernel_type="matern_5_2")
    training_time = model.train(X_train, Y_train)
    
    # 4. Predict
    mean_scaled, std_scaled, lower95_scaled, upper95_scaled, infer_time  = model.predict(X_test)
    
    # 5. Descale
    ground_truth, mean, std, lower95, upper95 = descale_data(
        Y_test.detach().cpu().numpy(), mean_scaled, std_scaled, lower95_scaled, upper95_scaled, 
        scaler_mean, scaler_scale)
    
    # 6. Compute evaluation metrics
    rmse = ErrorMetrics.RMSE(predictions=mean, observations=ground_truth)
    coverage_prob = ErrorMetrics.CoverageProbability(mean, lower95, upper95, ground_truth)
    quantile_coverage_error = ErrorMetrics.QuantileCoverageError(lower95, upper95, ground_truth)
    
    # 7. Visualize predictions and ground-truths
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = "ExactGP"
    predictions = np.concatenate([np.array(mean)[:,np.newaxis], np.array(std)[:,np.newaxis]], axis=1)
    plot_prediction(ground_truth, predictions, model_name, args.output_dir)
    
    # 8. Save prediction results and metrics
    hdf5_file = os.path.join(args.output_dir, "pred_and_gt.h5")
    with h5py.File(hdf5_file, 'w') as f:
        f.create_dataset('pred_mean', data=mean.astype(np.float64))
        f.create_dataset('gt', data=ground_truth.astype(np.float64))
    print(f"[ExactGP] wrote predictions and ground-truth → {hdf5_file}")
    metrics = dict(
        name="ExactGP",
        rmse=float(rmse),
        coverage_prob=float(coverage_prob),
        quantile_coverage_error=float(quantile_coverage_error),
        train_time=float(training_time),
        infer_time=float(infer_time)
    )
    with open(os.path.join(args.output_dir,"metrics.json"),"w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[ExactGP] metrics → {args.output_dir}/metrics.json")

if __name__=="__main__":
    main()