# High-dimensional output problem

# High-dim-output-specific config
_threshold = config["case_study"].get("preprocessing", {}).get("threshold", 5e-06)
_qoi = config["case_study"].get("qoi", "hmax")
_n_components = config["case_study"].get("dim_reduction", {}).get("n_components", 10)
_latent_dim = config["case_study"].get("dim_reduction", {}).get("latent_dim", 10)

# 1. Data Setup  (conditional on case study)
if CASE_STUDY == "environment_spill_function":

    rule data_setup:
        output:
            directory(f"{WORKDIR}/processed_data"),
        conda:
            "../envs/data_setup_env_spill_func.yml"
        params:
            script=f"{SCRIPTS}/high_dim_output/data_setup_env_spill_func.py",
            extra_args=build_optional_args(
                **{
                    "n-samples": get_dataset_param("n_samples"),
                    "seed": get_dataset_param("seed"),
                }
            ),
        shell:
            """
            python {params.script} --output-dir {output} {params.extra_args}
            """

elif CASE_STUDY in ["acheron", "synthetic_landslide"]:

    rule fetch_from_figshare:
        output:
            directory(f"{WORKDIR}/raw_data/figshare"),
        conda:
            "../envs/fetch_from_figshare.yml"
        params:
            fetch_cmds=generate_figshare_fetch_commands(),
        shell:
            """
            mkdir -p {output}
            cd {output}
            {params.fetch_cmds}
            """

    rule fetch_from_github:
        output:
            directory(f"{WORKDIR}/raw_data/github"),
        conda:
            "../envs/fetch_from_github.yml"
        params:
            fetch_cmds=generate_github_fetch_commands(),
        shell:
            """
            mkdir -p {output}
            cd {output}
            {params.fetch_cmds}
            """

    rule data_setup:
        input:
            figshare=f"{WORKDIR}/raw_data/figshare",
            github=f"{WORKDIR}/raw_data/github",
        output:
            directory(f"{WORKDIR}/processed_data"),
        conda:
            "../envs/data_setup_landslide.yml"
        params:
            script=f"{SCRIPTS}/high_dim_output/data_setup_landslide.py",
            qoi=_qoi,
        shell:
            """
            python {params.script} \
                --figshare-dir {input.figshare} \
                --github-dir {input.github} \
                --output-dir {output} \
                --qoi {params.qoi}
            """

else:
    raise ValueError(
        f"Unsupported case study for high_dim_output: {CASE_STUDY}"
    )


# 2. Preprocessing
rule preprocessing:
    input:
        f"{WORKDIR}/processed_data",
    output:
        directory(f"{WORKDIR}/data_tensors"),
    conda:
        "../envs/preprocessing.yml"
    params:
        script=f"{SCRIPTS}/high_dim_output/preprocessing.py",
        threshold=_threshold,
    shell:
        """
        [[ "$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE || true
        python {params.script} \
            --input-dir {input} \
            --output-dir {output} \
            --threshold {params.threshold}
        """


# 3. Model Evaluation
rule evaluate_ppgasp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_ppgasp"),
    conda:
        "../envs/evaluate_ppgasp.yml"
    benchmark:
        f"{WORKDIR}/benchmarks/evaluate_ppgasp.tsv"
    params:
        script=f"{SCRIPTS}/high_dim_output/evaluate_ppgasp.py",
        src_dir=SRC_DIR,
        threshold=_threshold,
        qoi=_qoi,
    shell:
        """
        export PYTHONPATH={params.src_dir}:${{PYTHONPATH:-}}
        python {params.script} \
            --input-dir {input} \
            --output-dir {output} \
            --threshold {params.threshold} \
            --qoi {params.qoi}
        """

rule evaluate_bigp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_bigp"),
    conda:
        gpu_env("evaluate_bigp")
    benchmark:
        f"{WORKDIR}/benchmarks/evaluate_bigp.tsv"
    params:
        script=f"{SCRIPTS}/high_dim_output/evaluate_bigp.py",
        src_dir=SRC_DIR,
        threshold=_threshold,
        qoi=_qoi,
        device=device_flag(),
    shell:
        """
        export PYTHONPATH={params.src_dir}:${{PYTHONPATH:-}}
        [[ "$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE || true
        python {params.script} \
            --input-dir {input} \
            --output-dir {output} \
            --threshold {params.threshold} \
            --qoi {params.qoi} \
            {params.device}
        """

rule evaluate_mtgp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_mtgp"),
    conda:
        gpu_env("evaluate_mtgp")
    benchmark:
        f"{WORKDIR}/benchmarks/evaluate_mtgp.tsv"
    params:
        script=f"{SCRIPTS}/high_dim_output/evaluate_mtgp.py",
        src_dir=SRC_DIR,
        threshold=_threshold,
        qoi=_qoi,
        device=device_flag(),
    shell:
        """
        export PYTHONPATH={params.src_dir}:${{PYTHONPATH:-}}
        [[ "$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE || true
        python {params.script} \
            --input-dir {input} \
            --output-dir {output} \
            --threshold {params.threshold} \
            --qoi {params.qoi} \
            {params.device}
        """

rule evaluate_pca_bigp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_pca_bigp"),
    conda:
        gpu_env("evaluate_pca_bigp")
    benchmark:
        f"{WORKDIR}/benchmarks/evaluate_pca_bigp.tsv"
    params:
        script=f"{SCRIPTS}/high_dim_output/evaluate_pca_bigp.py",
        src_dir=SRC_DIR,
        threshold=_threshold,
        n_components=_n_components,
        qoi=_qoi,
        device=device_flag(),
    shell:
        """
        export PYTHONPATH={params.src_dir}:${{PYTHONPATH:-}}
        [[ "$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE || true
        python {params.script} \
            --input-dir {input} \
            --output-dir {output} \
            --threshold {params.threshold} \
            --n-components {params.n_components} \
            --qoi {params.qoi} \
            {params.device}
        """

rule evaluate_pca_ppgasp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_pca_ppgasp"),
    conda:
        "../envs/evaluate_pca_ppgasp.yml"
    benchmark:
        f"{WORKDIR}/benchmarks/evaluate_pca_ppgasp.tsv"
    params:
        script=f"{SCRIPTS}/high_dim_output/evaluate_pca_ppgasp.py",
        src_dir=SRC_DIR,
        n_components=_n_components,
        threshold=_threshold,
        qoi=_qoi,
    shell:
        """
        export PYTHONPATH={params.src_dir}:${{PYTHONPATH:-}}
        python {params.script} \
            --input-dir {input} \
            --output-dir {output} \
            --n-components {params.n_components} \
            --threshold {params.threshold} \
            --qoi {params.qoi}
        """

rule evaluate_kpca_ppgasp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_kpca_ppgasp"),
    conda:
        "../envs/evaluate_kpca_ppgasp.yml"
    benchmark:
        f"{WORKDIR}/benchmarks/evaluate_kpca_ppgasp.tsv"
    params:
        script=f"{SCRIPTS}/high_dim_output/evaluate_kpca_ppgasp.py",
        src_dir=SRC_DIR,
        n_components=_n_components,
        qoi=_qoi,
        threshold=_threshold,
    shell:
        """
        export PYTHONPATH={params.src_dir}:${{PYTHONPATH:-}}
        python {params.script} \
            --input-dir {input} \
            --output-dir {output} \
            --n-components {params.n_components} \
            --qoi {params.qoi} \
            --threshold {params.threshold}
        """

rule evaluate_ae_ppgasp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_ae_ppgasp"),
    conda:
        gpu_env("evaluate_ae_ppgasp")
    benchmark:
        f"{WORKDIR}/benchmarks/evaluate_ae_ppgasp.tsv"
    params:
        script=f"{SCRIPTS}/high_dim_output/evaluate_ae_ppgasp.py",
        src_dir=SRC_DIR,
        threshold=_threshold,
        latent_dim=_latent_dim,
        qoi=_qoi,
        device=device_flag(),
    shell:
        """
        export PYTHONPATH={params.src_dir}:${{PYTHONPATH:-}}
        [[ "$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE || true
        python {params.script} \
            --input-dir {input} \
            --output-dir {output} \
            --threshold {params.threshold} \
            --latent-dim {params.latent_dim} \
            --qoi {params.qoi} \
            {params.device}
        """

rule evaluate_vae_ppgasp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_vae_ppgasp"),
    conda:
        gpu_env("evaluate_vae_ppgasp")
    benchmark:
        f"{WORKDIR}/benchmarks/evaluate_vae_ppgasp.tsv"
    params:
        script=f"{SCRIPTS}/high_dim_output/evaluate_vae_ppgasp.py",
        src_dir=SRC_DIR,
        threshold=_threshold,
        latent_dim=_latent_dim,
        qoi=_qoi,
        device=device_flag(),
    shell:
        """
        export PYTHONPATH={params.src_dir}:${{PYTHONPATH:-}}
        [[ "$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE || true
        python {params.script} \
            --input-dir {input} \
            --output-dir {output} \
            --threshold {params.threshold} \
            --latent-dim {params.latent_dim} \
            --qoi {params.qoi} \
            {params.device}
        """