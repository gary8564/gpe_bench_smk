rule evaluate_pca_rgasp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_pca_rgasp"),
    conda:
        conda_env("evaluate_pca_rgasp")
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
