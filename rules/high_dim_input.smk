"""
High-dimensional input problem
"""

# 1. Data Setup  (conditional on case study)
if CASE_STUDY == "synthetic_100d_function":

    rule data_setup:
        output:
            directory(f"{WORKDIR}/processed_data"),
        conda:
            "../envs/data_setup_100d_func.yml"
        params:
            script=f"{SCRIPTS}/high_dim_input/data_setup_100d_func.py",
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

elif CASE_STUDY == "tsunami_tokushima":

    rule fetch_from_zenodo:
        output:
            directory(f"{WORKDIR}/raw_data"),
        conda:
            "../envs/fetch_from_zenodo.yml"
        params:
            fetch_cmds=generate_zenodo_fetch_commands(),
        shell:
            """
            mkdir -p {output}
            cd {output}
            {params.fetch_cmds}
            for zipfile in *.zip; do
                if [ -f "$zipfile" ]; then
                    echo "Unzipping $zipfile..."
                    unzip -q "$zipfile"
                    rm "$zipfile"
                fi
            done
            """

    rule data_setup:
        input:
            f"{WORKDIR}/raw_data",
        output:
            directory(f"{WORKDIR}/processed_data"),
        conda:
            "../envs/data_setup_tsunami.yml"
        params:
            script=f"{SCRIPTS}/high_dim_input/data_setup_tsunami.py",
        shell:
            """
            python {params.script} --input-dir {input} --output-dir {output}
            """

else:
    raise ValueError(
        f"Unsupported case study for high_dim_input: {CASE_STUDY}"
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
        script=f"{SCRIPTS}/high_dim_input/preprocessing.py",
    shell:
        """
        [[ "$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE || true
        python {params.script} --input-dir {input} --output-dir {output}
        """


# 3. Model Evaluation
rule evaluate_exactgp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_exactgp"),
    conda:
        gpu_env("evaluate_exactgp")
    benchmark:
        f"{WORKDIR}/benchmarks/evaluate_exactgp.tsv"
    params:
        script=f"{SCRIPTS}/high_dim_input/evaluate_exactgp.py",
        src_dir=SRC_DIR,
        device=device_flag(),
    shell:
        """
        export PYTHONPATH={params.src_dir}:${{PYTHONPATH:-}}
        [[ "$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE || true
        python {params.script} \
            --input-dir {input} \
            --output-dir {output} \
            {params.device}
        """

rule evaluate_dkl:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_dkl"),
    conda:
        gpu_env("evaluate_dkl")
    benchmark:
        f"{WORKDIR}/benchmarks/evaluate_dkl.tsv"
    params:
        script=f"{SCRIPTS}/high_dim_input/evaluate_dkl.py",
        src_dir=SRC_DIR,
        device=device_flag(),
    shell:
        """
        export PYTHONPATH={params.src_dir}:${{PYTHONPATH:-}}
        [[ "$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE || true
        python {params.script} \
            --input-dir {input} \
            --output-dir {output} \
            {params.device}
        """

rule evaluate_rgasp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_rgasp"),
    conda:
        "../envs/evaluate_rgasp.yml"
    benchmark:
        f"{WORKDIR}/benchmarks/evaluate_rgasp.tsv"
    params:
        script=f"{SCRIPTS}/high_dim_input/evaluate_rgasp.R",
    shell:
        """
        Rscript {params.script} \
            --input-dir {input} \
            --output-dir {output}
        """

rule evaluate_pca_rgasp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_pca_rgasp"),
    conda:
        "../envs/evaluate_pca_rgasp.yml"
    benchmark:
        f"{WORKDIR}/benchmarks/evaluate_pca_rgasp.tsv"
    params:
        script=f"{SCRIPTS}/high_dim_input/evaluate_pca_rgasp.py",
        src_dir=SRC_DIR,
    shell:
        """
        export PYTHONPATH={params.src_dir}:${{PYTHONPATH:-}}
        [[ "$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE || true
        python {params.script} \
            --input-dir {input} \
            --output-dir {output}
        """