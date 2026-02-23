rule evaluate_ppgasp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_ppgasp"),
    conda:
        conda_env("evaluate_ppgasp")
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
