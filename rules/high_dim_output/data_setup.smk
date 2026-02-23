if CASE_STUDY == "environment_spill_function":

    rule data_setup:
        output:
            directory(f"{WORKDIR}/processed_data"),
        conda:
            conda_env("data_setup_env_spill_func")
        params:
            script=f"{SCRIPTS}/high_dim_output/data_setup_env_spill_func.py",
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

elif CASE_STUDY in ["acheron", "synthetic_landslide"]:

    rule fetch_from_figshare:
        output:
            directory(f"{WORKDIR}/raw_data/figshare"),
        conda:
            conda_env("fetch_from_figshare")
        params:
            fetch_cmds=generate_figshare_fetch_commands(),
        shell:
            """
            mkdir -p {output}
            cd {output}
            {params.fetch_cmds}
            """

    rule fetch_from_github:
        output:
            directory(f"{WORKDIR}/raw_data/github"),
        conda:
            conda_env("fetch_from_github")
        params:
            fetch_cmds=generate_github_fetch_commands(),
        shell:
            """
            mkdir -p {output}
            cd {output}
            {params.fetch_cmds}
            """

    rule data_setup:
        input:
            figshare=f"{WORKDIR}/raw_data/figshare",
            github=f"{WORKDIR}/raw_data/github",
        output:
            directory(f"{WORKDIR}/processed_data"),
        conda:
            conda_env("data_setup_landslide")
        params:
            script=f"{SCRIPTS}/high_dim_output/data_setup_landslide.py",
            qoi=_qoi,
        shell:
            """
            python {params.script} \
                --figshare-dir {input.figshare} \
                --github-dir {input.github} \
                --output-dir {output} \
                --qoi {params.qoi}
            """

else:
    raise ValueError(
        f"Unsupported case study for high_dim_output: {CASE_STUDY}"
    )
