if CASE_STUDY == "synthetic_100d_function":

    rule data_setup:
        output:
            directory(f"{WORKDIR}/processed_data"),
        conda:
            conda_env("data_setup_100d_func")
        params:
            script=f"{SCRIPTS}/high_dim_input/data_setup_100d_func.py",
            extra_args=build_optional_args(
                **{
                    "n-samples": get_dataset_param("n_samples"),
                    "seed": get_dataset_param("seed"),
                }
            ),
        shell:
            """
            python {params.script} --output-dir {output} {params.extra_args}
            """

elif CASE_STUDY == "tsunami_tokushima":

    rule fetch_from_zenodo:
        output:
            directory(f"{WORKDIR}/raw_data"),
        conda:
            conda_env("fetch_from_zenodo")
        params:
            fetch_cmds=generate_zenodo_fetch_commands(),
        shell:
            """
            mkdir -p {output}
            cd {output}
            {params.fetch_cmds}
            for zipfile in *.zip; do
                if [ -f "$zipfile" ]; then
                    echo "Unzipping $zipfile..."
                    unzip -q "$zipfile"
                    rm "$zipfile"
                fi
            done
            """

    rule data_setup:
        input:
            f"{WORKDIR}/raw_data",
        output:
            directory(f"{WORKDIR}/processed_data"),
        conda:
            conda_env("data_setup_tsunami")
        params:
            script=f"{SCRIPTS}/high_dim_input/data_setup_tsunami.py",
        shell:
            """
            python {params.script} --input-dir {input} --output-dir {output}
            """

else:
    raise ValueError(
        f"Unsupported case study for high_dim_input: {CASE_STUDY}"
    )
