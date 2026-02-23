rule evaluate_kpca_ppgasp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_kpca_ppgasp"),
    conda:
        conda_env("evaluate_kpca_ppgasp")
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
