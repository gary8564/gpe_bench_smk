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
