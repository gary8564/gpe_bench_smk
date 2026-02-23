rule evaluate_pca_ppgasp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_pca_ppgasp"),
    conda:
        conda_env("evaluate_pca_ppgasp")
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
