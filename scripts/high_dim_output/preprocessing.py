import argparse
import os
import numpy as np
import h5py
import shutil
from sklearn.preprocessing import StandardScaler

def zero_truncated_data(raw_data, threshold, valid_cols=None):
    """ 
    Preprocess the dataset to filter out the zeros so that GP emulators can be trained
        
    Args:
        raw_data (np.ndarray): Raw data to be preprocessed.
        threshold (int, float): Threshold value to define valid cells from simulations.
        valid_cols (list, optional): column numbers to extract. Defaults to None.
    
    Raises:
        TypeError: threshold must be a number
        ValueError: threshold cannot be negative
    
    Returns:
        training (np.ndarray): A data frame consisting of the vector outputs from simulations
        valid_cols (np.ndarray): An array consisting of the valid column names
    """
    if not isinstance(threshold, (int, float)):
        raise TypeError('threshold must be a number')
    if threshold < 0:
        raise ValueError('threshold cannot be negative')
    if not isinstance(raw_data, np.ndarray):
        raise TypeError('raw_data must be a np.ndarray')
    assert raw_data.ndim == 3, 'raw_data must be a 3D np.ndarray with shape (num_samples, rows, cols)'
    rows = raw_data.shape[1]
    cols = raw_data.shape[2]
    unstacked = raw_data.reshape(raw_data.shape[0], rows * cols)
    if valid_cols is None:
        valid_cols = np.where(unstacked >= threshold, 1, 0).sum(axis=0)
    indices = np.flatnonzero(valid_cols)
    nz_out = unstacked[:, indices]
    return nz_out, valid_cols, rows, cols

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True,
                   help="Directory containing train_X.npy, train_Y.npy, test_X.npy, test_Y.npy")
    p.add_argument("--output-dir", required=True,
                   help="Where to write standardization params as scaler.pkl and input/output tensor as .pt files")
    p.add_argument("--threshold", type=float, required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    X_train = np.load(os.path.join(args.input_dir,"train_X.npy"))
    Y_train = np.load(os.path.join(args.input_dir,"train_Y.npy"))
    X_test = np.load(os.path.join(args.input_dir,"test_X.npy"))
    Y_test = np.load(os.path.join(args.input_dir,"test_Y.npy"))
    
    # 1. Filter out the zeros
    Y_train, valid_cols, rows, cols = zero_truncated_data(Y_train, args.threshold)
    Y_test, _, rows, cols = zero_truncated_data(Y_test, args.threshold, valid_cols)

    # 2. Standardization
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    X_train_scaled = input_scaler.fit_transform(X_train)
    Y_train_scaled = output_scaler.fit_transform(Y_train)
    X_test_scaled = input_scaler.transform(X_test)
    Y_test_scaled = output_scaler.transform(Y_test)
    
    # Optional: Copy background image to pass on for later visualization
    bg_src = os.path.join(args.input_dir, "background.tif")
    bg_dest = None
    if os.path.exists(bg_src):
        bg_dest = os.path.join(args.output_dir, "background.tif")
        shutil.copy2(bg_src, bg_dest)
        
    # 3. Save to HDF5 (language-agnostic format)
    hdf5_file = os.path.join(args.output_dir, "data.h5")
    with h5py.File(hdf5_file, 'w') as f:
        # Save datasets
        f.create_dataset('train_X', data=X_train_scaled.astype(np.float64))
        f.create_dataset('train_Y', data=Y_train_scaled.astype(np.float64))
        f.create_dataset('test_X', data=X_test_scaled.astype(np.float64))
        f.create_dataset('test_Y', data=Y_test_scaled.astype(np.float64))
        
        # Save standardization parameters
        f.create_dataset('output_scaler_mean', data=output_scaler.mean_.astype(np.float64))
        f.create_dataset('output_scaler_scale', data=output_scaler.scale_.astype(np.float64))
        
        # Save metadata
        f.create_dataset("valid_indices", data=valid_cols.astype(np.int64))
        f.attrs["rows"] = int(rows)
        f.attrs["cols"] = int(cols)
        f.attrs["threshold"] = float(args.threshold)
        f.attrs["background_img_path"] = bg_dest if bg_dest else ""

    print(f"[preprocess] wrote HDF5 data → {hdf5_file}")

if __name__=="__main__":
    main()




