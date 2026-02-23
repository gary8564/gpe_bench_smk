_threshold = config["case_study"].get("preprocessing", {}).get("threshold", 5e-06)
_qoi = config["case_study"].get("qoi", "hmax")
_n_components = config["case_study"].get("dim_reduction", {}).get("n_components", 10)
_latent_dim = config["case_study"].get("dim_reduction", {}).get("latent_dim", 10)
