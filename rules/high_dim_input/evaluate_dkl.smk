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
