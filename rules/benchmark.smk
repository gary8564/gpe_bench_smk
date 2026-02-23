rule benchmark_metrics:
    input:
        results=[f"{WORKDIR}/results_{model}" for model in MODELS],
    output:
        directory(f"{WORKDIR}/benchmark_results"),
    conda:
        conda_env("benchmark_metrics")
    params:
        script=f"{SCRIPTS}/benchmark_metrics.py",
        src_dir=SRC_DIR,
        problem_type=PROBLEM_TYPE,
        qoi_flag=(
            f"--qoi {config['case_study']['qoi']}"
            if config["case_study"].get("qoi")
            else ""
        ),
        metrics_json=lambda wildcards, input: json.dumps(list(input.results)),
    shell:
        """
        export PYTHONPATH={params.src_dir}:${{PYTHONPATH:-}}
        python {params.script} \
            --metrics-paths '{params.metrics_json}' \
            --output-dir {output} \
            --problem-type {params.problem_type} \
            {params.qoi_flag}
        """
