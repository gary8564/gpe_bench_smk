configfile: "config.yaml"


include: "rules/config.smk"
include: "rules/utils.smk"
include: "rules/data_fetching.smk"


rule all:
    input:
        f"{WORKDIR}/benchmark_results",


if PROBLEM_TYPE == "high_dim_input":
    include: "rules/high_dim_input.smk"
elif PROBLEM_TYPE == "high_dim_output":
    include: "rules/high_dim_output.smk"
else:
    raise ValueError(f"Unsupported problem_type: {PROBLEM_TYPE}")

include: "rules/benchmark.smk"
