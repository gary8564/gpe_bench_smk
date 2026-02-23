import os
import json


CASE_STUDY = config["case_study"]["name"]
PROBLEM_TYPE = config["case_study"]["problem_type"]
OUTDIR = config.get("outdir", "results")
USE_GPU = config.get("use_gpu", False)
DATASETS = config.get("datasets", {})

WORKDIR = os.path.join(OUTDIR, CASE_STUDY)
SCRIPTS = os.path.join(workflow.basedir, "scripts")
SRC_DIR = os.path.join(workflow.basedir, "src")


def get_models():
    """Determine which models to evaluate based on problem type and case study."""
    if PROBLEM_TYPE == "high_dim_input":
        if CASE_STUDY == "synthetic_100d_function":
            return ["exactgp", "dkl", "rgasp", "pca_rgasp"]
        elif CASE_STUDY == "tsunami_tokushima":
            return ["exactgp", "dkl", "pca_rgasp"]
        else:
            raise ValueError(
                f"Unsupported case study for high_dim_input: {CASE_STUDY}"
            )
    elif PROBLEM_TYPE == "high_dim_output":
        return [
            "ppgasp", "pca_ppgasp", "kpca_ppgasp", "ae_ppgasp",
            "vae_ppgasp", "bigp", "pca_bigp", "mtgp",
        ]
    else:
        raise ValueError(f"Unsupported problem_type: {PROBLEM_TYPE}")

MODELS = get_models()
