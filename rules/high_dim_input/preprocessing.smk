rule preprocessing:
    input:
        f"{WORKDIR}/processed_data",
    output:
        directory(f"{WORKDIR}/data_tensors"),
    conda:
        conda_env("preprocessing")
    params:
        script=f"{SCRIPTS}/high_dim_input/preprocessing.py",
    shell:
        """
        [[ "$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE || true
        python {params.script} --input-dir {input} --output-dir {output}
        """
