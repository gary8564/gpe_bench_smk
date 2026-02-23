import argparse
import os
import json
import numpy as np
import h5py
from high_dim_gp.emulator.bigp import PCA_BiGP
from high_dim_gp.dr.pca import OutputDimReducer, LinearPCA
from high_dim_gp.utils.error_metrics import ErrorMetrics
from high_dim_gp.utils.plot import plot_prediction, plot_residuals, plot_error_maps

def main():
    # 1. Parse arguments
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-components", type=int, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--optim", default="adamw")
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
        
    # 4. Train
    output_reducer = OutputDimReducer(LinearPCA(n_components=args.n_components))
    emulator = PCA_BiGP(output_reducer, device=args.device, kernel_type='matern_5_2')
    X_train, Y_train_reduced, X_test, Y_test_reduced = emulator.preprocess_dim_reduction(X_train, Y_train, X_test, Y_test)
    training_time = emulator.train(X_train,
                                   Y_train_reduced,
                                   num_epochs=args.num_epochs,
                                   lr=args.lr,
                                   optim=args.optim)
    
    # 5. Predict
    mean, std, _, _, infer_time = emulator.predict(X_test)
    
    # 6. Postprocessing: reconstruct back to original space
    mean_original, std_original, lower_CI, upper_CI = emulator.postprocess_invert_back(mean, std, scaler_mean, scaler_scale)
    
    # 7. Descale and filter out predictive values below threshold
    ground_truth = Y_test * scaler_scale + scaler_mean
    ground_truth = np.where(ground_truth < args.threshold, 0, ground_truth)
    predictions_mean = np.where(mean_original < args.threshold, 0, mean_original)
    predictions_lower = np.where(mean_original < args.threshold, 0, lower_CI)
    predictions_upper = np.where(mean_original < args.threshold, 0, upper_CI)
    predictions_std = np.where(mean_original < args.threshold, 0, std_original)
        
    # 8. Compute evaluation metrics
    rmse = ErrorMetrics.RMSE(predictions=predictions_mean, observations=ground_truth)
    coverage_prob = ErrorMetrics.CoverageProbability(predictions_mean, predictions_lower, predictions_upper, ground_truth)
    quantile_coverage_error = ErrorMetrics.QuantileCoverageError(predictions_lower, predictions_upper, ground_truth)
    
    # 9. Visualize predictions and ground-truths
    model_name = "PCA-BiGP"
    predictions = np.dstack((predictions_mean, predictions_std))
    plot_prediction(ground_truth, predictions, model_name, output_dir) 
    plot_residuals(ground_truth, predictions, model_name, output_dir)
    plot_error_maps(ground_truth, predictions_mean, output_img_params, model_name, output_dir, args.qoi)
    
    # 10. Save metrics
    metrics = dict(
        name="PCA-BiGP",
        rmse=float(rmse),
        coverage_prob=float(coverage_prob),
        quantile_coverage_error=float(quantile_coverage_error),
        train_time=float(training_time),
        infer_time=float(infer_time)
    )
    with open(os.path.join(output_dir,"metrics.json"),"w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[PCA-BiGP] metrics → {output_dir}/metrics.json")
    
    # 11. Save predictions and ground-truth for later benchmarking analysis
    hdf5_file = os.path.join(output_dir, "pred_and_gt.h5")
    with h5py.File(hdf5_file, 'w') as f:
        f.create_dataset('pred_mean', data=predictions_mean.astype(np.float64))
        f.create_dataset('gt', data=ground_truth.astype(np.float64))
    print(f"[PCA-BiGP] wrote predictions and ground-truth → {hdf5_file}")

if __name__=="__main__":
    main()