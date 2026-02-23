import argparse
import os
import json
import ast
import numpy as np
import pandas as pd
import h5py
from high_dim_gp.utils.plot import plot_output_distribution, plot_rmse_vs_train_time, plot_coverage_prob, plot_pca_zero_output_hist

def load_metrics(metrics_path: str) -> dict:
    if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"[benchmark_metrics] Missing metrics file: {metrics_path}")
    with open(metrics_path, "r") as f:
        try:
            metrics = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"[benchmark_metrics] Error decoding JSON: {metrics_path}: {e}")
    return metrics

def load_predictions(hdf5_filepath: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(hdf5_filepath):
        raise FileNotFoundError(f"[benchmark_metrics] Missing predictions file: {hdf5_filepath}")
    with h5py.File(hdf5_filepath, 'r') as f:
        predictions = np.ravel(f['pred_mean'][:])
        ground_truths = np.ravel(f['gt'][:])
    return predictions, ground_truths

def main():
    # 1. Parse arguments 
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-paths", required=True, type=ast.literal_eval,
                   help="A list of directory paths containing metrics.json files to benchmark")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--problem-type", required=True, choices=["high_dim_output", "high_dim_input"])
    p.add_argument("--qoi", required=False, default=None, choices=["hmax", "vmax", "cmax"])
    args = p.parse_args()
    if args.problem_type == "high_dim_output" and not args.qoi:
        raise ValueError("--qoi is required when --problem-type is 'high_dim_output'")

    # 2. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Load metrics and output values from each path
    if not isinstance(args.metrics_paths, list) or len(args.metrics_paths) == 0:
        raise ValueError("[benchmark_metrics] Error: --metrics-paths must be a non-empty list of directory paths")
    
    benchmark_metrics = {}
    df_output_values = []
    model_predictions = []
    model_names = []
    for i, p in enumerate(args.metrics_paths):
        if not isinstance(p, str):
            raise ValueError(f"[benchmark_metrics] Error: item {i} must be a string path, got {type(p)}")
        # Load metrics and save in benchmark_metrics
        metrics_path = os.path.join(p, "metrics.json")
        metrics = load_metrics(metrics_path)
        model_name = metrics.get("name")
        if model_name is None:
            raise ValueError(f"[benchmark_metrics] Error: item {i} missing 'name' field")
        benchmark_metrics[model_name] = metrics
        model_names.append(model_name)
        
        # Load predictions and ground-truths and save in model_predictions and df_output_values
        predictions, ground_truths = load_predictions(os.path.join(p, "pred_and_gt.h5"))
        model_predictions.append(predictions)
        df_output_values.append(pd.DataFrame({"Model": model_name, "Source": "Ground-truth", "Value": ground_truths}))
        df_output_values.append(pd.DataFrame({"Model": model_name, "Source": "Prediction",   "Value": predictions}))
    if len(df_output_values) > 0:
        df_output_values = pd.concat(df_output_values, ignore_index=True)
    else:
        raise ValueError("[benchmark_metrics] Error: Output values are empty.")

    # 4. Save evaluation metrics comparison as a csv 
    benchmark_data = []
    for model_name, metrics in benchmark_metrics.items():
        benchmark_row = {
            'rmse': metrics["rmse"], 
            'train_time': metrics["train_time"], 
            'infer_time': metrics["infer_time"]
        }
        benchmark_data.append(benchmark_row)
    df = pd.DataFrame(benchmark_data, index=model_names)
    csv_path = os.path.join(args.output_dir, "comparison.csv")
    df.to_csv(csv_path)
    print(f"[benchmark_metrics] Saved comparison → {csv_path}")
    
    # 5. Visualize RMSE vs Training Time scatter plot
    plot_rmse_vs_train_time(benchmark_metrics, args.output_dir)
    print("[benchmark_metrics] Generated RMSE vs Training Time scatter plot")
    
    # 6. Visualize output distribution and coverage probability
    plot_output_distribution(df_output_values, args.output_dir, args.qoi)
    plot_coverage_prob(benchmark_metrics, args.output_dir)

    # 7. (Optional for high-dim-output) Visualize zero-output distribution
    if args.problem_type == "high_dim_output":
        plot_pca_zero_output_hist(model_predictions, model_names, ground_truths, args.output_dir)
if __name__=="__main__":
    main()
