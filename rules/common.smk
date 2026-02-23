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

def gpu_env(base_name):
    """Return CUDA conda env path when GPU is enabled, otherwise CPU env."""
    suffix = "_cuda" if USE_GPU else ""
    return f"../envs/{base_name}{suffix}.yml"

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


# Data-fetching command generators
def generate_zenodo_fetch_commands():
    ds = DATASETS[CASE_STUDY]
    base_url = ds["base_url"]
    cmds = []
    for f in ds["files"]:
        cmds.append(f'echo "Downloading {f}..."')
        cmds.append(
            f'wget -q --show-progress -O "{f}" '
            f'"{base_url}/files/{f}?download=1"'
        )
    return "\n".join(cmds)

def generate_figshare_fetch_commands():
    ds = DATASETS[CASE_STUDY]
    articles = ds.get("figshare", {}).get("articles", [])
    cmds = []
    for art in articles:
        art_id = art["id"]
        dest = art["dest"]
        cmds.append(f'mkdir -p "{dest}"')
        cmds.append(f'echo "Fetching Figshare article {art_id} -> {dest}"')
        cmds.append(
            f'curl -sS "https://api.figshare.com/v2/articles/{art_id}" \\\n'
            f"  | jq -r '.files[]? | [.download_url, .name] | @tsv' \\\n"
            f"  | while IFS=$'\\t' read -r url name; do\n"
            f'      echo "Downloading $name"\n'
            f'      curl -L --fail --retry 3 --retry-delay 2 '
            f'-o "{dest}/$name" "$url"\n'
            f'      case "$name" in\n'
            f"        *.zip|*.ZIP|*.Zip)\n"
            f'          unzip -q -o "{dest}/$name" -d "{dest}"\n'
            f'          rm -f "{dest}/$name"\n'
            f"          ;;\n"
            f"      esac\n"
            f"    done"
        )
    return "\n".join(cmds)

def generate_github_fetch_commands():
    ds = DATASETS[CASE_STUDY]
    files = ds.get("github", {}).get("files", [])
    cmds = []
    for f in files:
        url = f["url"]
        raw_url = (
            url.replace("github.com", "raw.githubusercontent.com")
               .replace("/blob/", "/")
        )
        dest = f["dest"]
        filename = raw_url.split("/")[-1]
        cmds.append(f'mkdir -p "{dest}"')
        cmds.append(f'echo "Downloading {filename} -> {dest}"')
        cmds.append(
            f'curl -L --fail --retry 3 --retry-delay 2 '
            f'-o "{dest}/{filename}" "{raw_url}"'
        )
    return "\n".join(cmds)