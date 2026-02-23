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
