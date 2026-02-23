import numpy as np
import argparse
import os
from psimpy.sampler.latin import LHS
from sklearn.model_selection import train_test_split

M_MIN_MAX = [7, 13]
D_MIN_MAX = [0.02, 0.12]
INPUT_DOMAIN = np.array([M_MIN_MAX, D_MIN_MAX])

def environ(x, s=None, t=None, return_grid=False):
    """
    Environmental model function. (Surjanovic & Bingham)

    Parameters
    ----------
    x : shape (2,) or (n, 2), with each row is [M, D].
    s : vector of locations (optional), with default value [0.5, 1, 1.5, 2, 2.5]
    t : vector of times (optional), with default value [0.3, 0.6, ..., 60.0].
    return_grid : bool, optional
        If True, returns shape (n, ds, dt) (or (ds, dt) for single input).
        If False (default), returns flattened (n, ds*dt) (or (ds*dt,) for single input).
    tau : specified time of the second spill, optional

    Returns
    -------
    y : np.ndarray
        row vector of scaled concentrations of the pollutant at the space-time vectors (s, t).
        [y(s1,t1), y(s1,t2), ..., y(s1,tdt), y(s2,t1), ...] if return_grid is False.
    """
    x = np.asarray(x, dtype=float)
    single = (x.ndim == 1)
    if single:
        x = x[None, :]
        
    if x.shape[1] != 2 :
        raise ValueError("`x` must have shape (n, 2)")

    if t is None:
        t = np.linspace(0.3, 60.0, 200, dtype=float)
    else:
        t = np.asarray(t, dtype=float)
    if s is None:
        s = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)
    else:
        s = np.asarray(s, dtype=float)
        
    n = x.shape[0]
    ds = s.size
    dt = t.size

    # Shapes for broadcasting
    M   = x[:, 0][:, None, None]
    D   = x[:, 1][:, None, None]

    s   = s[None, :, None]
    t   = t[None, None, :]

    # C = M / sqrt(4*pi*D*t) * exp(-s^2 / (4*D*t))
    C = (M / np.sqrt(4.0 * np.pi * D * t)) * np.exp(-(s**2) / (4.0 * D * t))

    Y = np.sqrt(4.0 * np.pi) * C
    
    if not return_grid:
        Y = Y.reshape(n, ds * dt)

    return Y[0] if single else Y


def generate_toy_example(n_samples=250, seed=None):
    # Define LHS sampler
    sampler = LHS(ndim=2, bounds=INPUT_DOMAIN)

    # Generate samples to obtain input X
    nsamples = n_samples
    X = sampler.sample(nsamples)
    
    # Evaluate function to obtain output Y
    s_grid = np.linspace(0.0, 3.0, 50) 
    t_grid = np.linspace(0.3, 20, 200)  
    Y = environ(X, s=s_grid, t=t_grid, return_grid=True)
    
    return X, Y


def main():
    # 1. Parse arguments
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True,
                   help="Store generated train_X.npy, train_Y.npy, test_X.npy, test_Y.npy")
    p.add_argument("--n-samples", type=int, default=500,
                   help="Number of samples to generate")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    args = p.parse_args()
    
    # 2. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Generate synthetic data
    X, Y = generate_toy_example(n_samples=args.n_samples, seed=args.seed)
    
    # 4. Train/Test Split
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 5. Save data
    np.save(os.path.join(args.output_dir, "train_X.npy"), train_X)
    np.save(os.path.join(args.output_dir, "train_Y.npy"), train_Y)
    np.save(os.path.join(args.output_dir, "test_X.npy"), test_X)
    np.save(os.path.join(args.output_dir, "test_Y.npy"), test_Y)
    print(f"""[data_setup_toy_example] saved 
          input → {args.output_dir}/train_X.npy (shape: {train_X.shape})
                  {args.output_dir}/test_X.npy (shape: {test_X.shape})
          output → {args.output_dir}/train_Y.npy (shape: {train_Y.shape})
                   {args.output_dir}/test_Y.npy (shape: {test_Y.shape})""")

if __name__ == "__main__":
    main() 