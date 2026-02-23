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
