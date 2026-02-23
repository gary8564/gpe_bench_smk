rule evaluate_rgasp:
    input:
        f"{WORKDIR}/data_tensors",
    output:
        directory(f"{WORKDIR}/results_rgasp"),
    conda:
        conda_env("evaluate_rgasp")
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
