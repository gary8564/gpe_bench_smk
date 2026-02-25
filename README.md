# Evaluation Pipeline for Gaussian Process Emulators with High-dimensional Dataset

Gaussian Processes (GPs) are suffering from the "curse of the dimensionality". As input or output dimension grows up, the computation becomes intractable. This project aims to explore the state-of-the-art research of dimensionality reduction in Gaussian Process emulation. In this repository, a workflow using **Snakemake** (a orchestrated workflow management framework) is constructed to facilitate benchmarking different Gaussian Process models on high-dimensional input/output problems with minimal efforts.

## Workflow

The pipeline follows a 4-step workflow:

1. **Data Setup**: Generate synthetic data or fetch and process real-world data
2. **Preprocessing**: Standardize, split, and save data in **HDF5 format** (language-agnostic)
3. **Model Evaluation**: Train and evaluate **GP models in parallel**:

   **High-dimensional input models:**
   - ExactGP (Python/GPyTorch)
   - DKL (Python/GPyTorch)
   - RGaSP (R/RobustGaSP)
   - PCA-RGaSP (R/RobustGaSP + PCA)

   **High-dimensional output models:**
   - PPGaSP, PCA-PPGaSP, kPCA-PPGaSP (R/RobustGaSP)
   - AE-PPGaSP, VAE-PPGaSP (PyTorch + R/RobustGaSP)
   - BiGP, PCA-BiGP, MTGP (Python/GPyTorch)

4. **Benchmark Metrics**: Compare model performance and save results

## Datasets

| Name                         | Type              | Problem         | Description                                                                     |
| ---------------------------- | ----------------- | --------------- | ------------------------------------------------------------------------------- |
| `synthetic_100d_function`    | Synthetic         | High-dim input  | 100D synthetic test function                                                    |
| `tsunami_tokushima`          | Zenodo            | High-dim input  | Tsunami inundation surrogate model ([DOI](https://zenodo.org/records/15093228)) |
| `environment_spill_function` | Synthetic         | High-dim output | Environment spill toy model                                                     |
| `acheron`                    | Figshare + GitHub | High-dim output | Acheron rock avalanche                                                          |
| `synthetic_landslide`        | Figshare + GitHub | High-dim output | Synthetic landslide simulation                                                  |

## Folder Structure

```
.
├── Snakefile                    # Main workflow orchestration
├── config.yaml                  # All configurable parameters
├── pyproject.toml               # Python package definition (src/high_dim_gp)
├── rules/                       # Modular Snakemake rules
│   ├── config.smk               # Shared config variables, paths, model selection
│   ├── utils.smk                # Reusable helpers (conda_env, gpu_env, device_flag, etc.)
│   ├── data_fetching.smk        # Data fetch command generators (Zenodo/Figshare/GitHub)
│   ├── benchmark.smk            # Performance comparison and reporting
│   ├── high_dim_input.smk       # Hub: includes all high-dim input step files
│   ├── high_dim_input/          # Per-step rules
│   │   ├── data_setup.smk
│   │   ├── preprocessing.smk
│   │   ├── evaluate_exactgp.smk
│   │   ├── evaluate_dkl.smk
│   │   ├── evaluate_rgasp.smk
│   │   └── evaluate_pca_rgasp.smk
│   ├── high_dim_output.smk      # Hub: includes all high-dim output step files
│   └── high_dim_output/         # Per-step rules
│       ├── config.smk           # Problem-type-specific params (threshold, qoi, n_components)
│       ├── data_setup.smk
│       ├── preprocessing.smk
│       ├── evaluate_ppgasp.smk
│       ├── evaluate_bigp.smk
│       ├── evaluate_mtgp.smk
│       ├── evaluate_pca_bigp.smk
│       ├── evaluate_pca_ppgasp.smk
│       ├── evaluate_kpca_ppgasp.smk
│       ├── evaluate_ae_ppgasp.smk
│       └── evaluate_vae_ppgasp.smk
├── envs/                        # Per-rule Conda environment specifications
├── scripts/                     # Implementation scripts (Python & R)
│   ├── benchmark_metrics.py
│   ├── high_dim_input/
│   └── high_dim_output/
├── src/                         # Shared Python package (high_dim_gp)
│   └── high_dim_gp/
├── profiles/                    # Snakemake execution profiles
│   ├── local/config.yaml
│   └── slurm/config.yaml
└── results/                     # Pipeline outputs (auto-generated)
```

## Prerequisites

1. **Conda Environment Manager**:
   - [**conda**](https://www.anaconda.com/docs/getting-started/miniconda/install) or [**micromamba**](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
   - [**conda-lock**](https://conda.github.io/conda-lock/getting_started/#__tabbed_1_2)

2. [**Snakemake**](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html) (>= 8.0)

   ```bash
   conda install -c conda-forge -c bioconda snakemake
   ```

   Or with pip:

   ```bash
   pip install snakemake
   ```

> [!NOTE]
> If SLURM cluster execution is needed, follows the additional installation of [**SLURM executor plugin**]():
>
> ```bash
>    mamba install -c conda-forge -c bioconda snakemake-executor-plugin-slurm
> ```
>
> Or with pip:
>
> ```bash
> pip install snakemake-executor-plugin-slurm
> ```

## Quick Start

1. Clone repository:

   ```bash
   git clone https://github.com/gary8564/snakemake_demo.git
   cd snakemake_demo
   ```

2. Run the default pipeline (synthetic 100D function, local):
   ```bash
   snakemake --profile profiles/local
   ```

## Usage

### Configuration

All parameters are defined in `config.yaml`:

```yaml
outdir: results
use_gpu: false
case_study:
  name: synthetic_100d_function # which case study to run
  problem_type: high_dim_input # high_dim_input | high_dim_output
```

For high-dimensional output problems, additional parameters are required:

```yaml
case_study:
  name: environment_spill_function
  problem_type: high_dim_output
  qoi: cmax # quantity of interest: hmax, vmax, or cmax
  preprocessing:
    threshold: 5e-06 # zero-truncation threshold
  dim_reduction:
    n_components: 10 # PCA/kPCA components
    latent_dim: 10 # AE/VAE latent dimension
```

### Override Config via Command Line

Snakemake allows overriding config values at the command line:

```bash
snakemake --profile profiles/local \
  --config case_study="{'name': 'tsunami_tokushima', 'problem_type': 'high_dim_input'}"
```

### GPU Acceleration

Set `use_gpu: true` in `config.yaml`. This selects CUDA-enabled Conda
environments for GPU-capable models (ExactGP, DKL, BiGP, MTGP, AE/VAE-PPGaSP).

### SLURM Cluster Execution

```bash
snakemake --profile profiles/slurm
```

### Custom Datasets

Extend the `datasets` section in `config.yaml`:

```yaml
datasets:
  my_new_study:
    source: "zenodo"
    description: "Description of your dataset"
    doi: "10.5281/zenodo.XXXXXXX"
    base_url: "https://zenodo.org/records/XXXXXXX"
    files:
      - "data_file1.csv"
      - "data_file2.zip"
```

### Dry Run

Preview the execution plan without running anything:

```bash
snakemake --profile profiles/local -n
```

### Command options

```bash
snakemake --profile profiles/local --dag \| dot -Tpng > dag.png # Generate DAG image
snakemake --profile profiles/local --forceall # Force re-run everything
snakemake --profile profiles/local --until preprocessing # Run up to a specific rule |
snakemake --profile profiles/local -R evaluate_exactgp # Re-run a specific rule |
snakemake --profile profiles/local --summary # Show output file status |
```

## Built-in Benchmarking

Each model evaluation rule uses Snakemake's [`benchmark`](https://snakemake.readthedocs.io/en/stable/tutorial/additional_features.html#benchmarking)
directive to automatically capture wall clock time, CPU time, and peak memory
usage. After a run, TSV files are written to:

```
results/<case_study>/benchmarks/
├── evaluate_exactgp.tsv        # high-dim input models
├── evaluate_dkl.tsv
├── evaluate_rgasp.tsv
├── evaluate_pca_rgasp.tsv
├── evaluate_ppgasp.tsv         # high-dim output models
├── evaluate_bigp.tsv
├── evaluate_mtgp.tsv
├── ...
```

## Comparison between Nextflow and Snakemake

| Aspect             | Nextflow                              | Snakemake                                                                       |
| ------------------ | ------------------------------------- | ------------------------------------------------------------------------------- |
| Workflow language  | Groovy DSL                            | Python-based                                                                    |
| Paradigm           | Channel-based (push; top-down)        | File-based DAG (pull; bottom-up)                                                |
| Config             | `.config`                             | `.yaml`                                                                         |
| Modules            | `modules/*.nf` (one process per file) | `rules/high_dim_input/*.smk`, `rules/high_dim_output/*.smk` (one rule per file) |
| Intermediate files | Hidden in `work/` directory           | Visible in output directory                                                     |
