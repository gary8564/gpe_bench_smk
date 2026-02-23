import numpy as np
import rasterio
import argparse
import os
import shutil

def load_dataset(input_filepath, output_filepath):
    input_data = np.genfromtxt(input_filepath, delimiter=',', skip_header=1)
    with rasterio.open(output_filepath) as output:
        rows = output.height
        cols = output.width
        size = output.count
        output_data = np.zeros((size, rows, cols))
        for sim in range(size):
            output_data[sim, :] = output.read(sim + 1).reshape(1, rows, cols)  
    return input_data, output_data

def main():
    # 1. Parse arguments
    p = argparse.ArgumentParser()
    p.add_argument("--figshare-dir", required=True,
                   help="Directory containing Figshare data (output data)")
    p.add_argument("--github-dir", required=True,
                   help="Directory containing GitHub data (input data)")
    p.add_argument("--output-dir", required=True,
                   help="Store processed train_X.npy, train_Y.npy, test_X.npy, test_Y.npy")
    p.add_argument("--qoi", required=True,
                   help="Quantity of interest",
                   choices=["hmax", "vmax"])
    args = p.parse_args()
    
    # 2. Check if input directories exist
    if not os.path.exists(args.figshare_dir):
        raise ValueError(f"Figshare directory does not exist: {args.figshare_dir}")
    if not os.path.exists(args.github_dir):
        raise ValueError(f"GitHub directory does not exist: {args.github_dir}")
    
    # 3. Construct file paths
    train_input_filepath = os.path.join(args.github_dir, "train", "input", "synth_emulator.csv")
    train_output_filepath = os.path.join(args.figshare_dir, "train", "output", args.qoi + "_stack.tif") 
    test_input_filepath = os.path.join(args.github_dir, "test", "input", "synth_validation_emulator.csv")
    test_output_filepath = os.path.join(args.figshare_dir, "test", "output", args.qoi + "_stack.tif") 
    hill_path = os.path.join(args.github_dir, "background", "hillshade_acheron.tif") 
    if not os.path.exists(hill_path):
        hill_path = None
        
    # 4. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 5. Copy background image
    if hill_path is not None:
        shutil.copy(hill_path, os.path.join(args.output_dir, "background.tif"))
    
    # 6. Load synthetic data
    train_X, train_Y = load_dataset(train_input_filepath, train_output_filepath)
    test_X, test_Y = load_dataset(test_input_filepath, test_output_filepath)
    
    # 7. Save data
    np.save(os.path.join(args.output_dir, "train_X.npy"), train_X)
    np.save(os.path.join(args.output_dir, "train_Y.npy"), train_Y)
    np.save(os.path.join(args.output_dir, "test_X.npy"), test_X)
    np.save(os.path.join(args.output_dir, "test_Y.npy"), test_Y)
    print(f"""[data_setup_synthetic] saved 
          input → {args.output_dir}/train_X.npy (shape: {train_X.shape})
                  {args.output_dir}/test_X.npy (shape: {test_X.shape})
          output → {args.output_dir}/train_Y.npy (shape: {train_Y.shape})
                   {args.output_dir}/test_Y.npy (shape: {test_Y.shape})""")
if __name__ == "__main__":
    main()