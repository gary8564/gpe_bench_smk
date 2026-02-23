def conda_env(name):
    """Return the absolute path to a conda environment YAML."""
    return os.path.join(workflow.basedir, "envs", f"{name}.yml")

def gpu_env(name):
    """Return CUDA conda env when GPU is enabled, otherwise the CPU variant."""
    suffix = "_cuda" if USE_GPU else ""
    return conda_env(f"{name}{suffix}")

def device_flag():
    """Return the --device CLI flag value."""
    return "--device cuda" if USE_GPU else "--device cpu"

def get_dataset_param(key, default=None):
    """Look up a parameter from the current dataset's config."""
    return DATASETS.get(CASE_STUDY, {}).get("parameters", {}).get(key, default)

def build_optional_args(**kwargs):
    """Build a CLI arg string from key-value pairs, skipping None values."""
    parts = []
    for flag, value in kwargs.items():
        if value is not None:
            parts.append(f"--{flag} {value}")
    return " ".join(parts)
