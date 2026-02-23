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
