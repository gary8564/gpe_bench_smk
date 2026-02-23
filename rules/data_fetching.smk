def generate_zenodo_fetch_commands():
    ds = DATASETS[CASE_STUDY]
    base_url = ds["base_url"]
    cmds = []
    for f in ds["files"]:
        cmds.append(f'echo "Downloading {f}..."')
        cmds.append(
            f'wget -q --show-progress -O "{f}" '
            f'"{base_url}/files/{f}?download=1"'
        )
    return "\n".join(cmds)

def generate_figshare_fetch_commands():
    ds = DATASETS[CASE_STUDY]
    articles = ds.get("figshare", {}).get("articles", [])
    cmds = []
    for art in articles:
        art_id = art["id"]
        dest = art["dest"]
        cmds.append(f'mkdir -p "{dest}"')
        cmds.append(f'echo "Fetching Figshare article {art_id} -> {dest}"')
        cmds.append(
            f'curl -sS "https://api.figshare.com/v2/articles/{art_id}" \\\n'
            f"  | jq -r '.files[]? | [.download_url, .name] | @tsv' \\\n"
            f"  | while IFS=$'\\t' read -r url name; do\n"
            f'      echo "Downloading $name"\n'
            f'      curl -L --fail --retry 3 --retry-delay 2 '
            f'-o "{dest}/$name" "$url"\n'
            f'      case "$name" in\n'
            f"        *.zip|*.ZIP|*.Zip)\n"
            f'          unzip -q -o "{dest}/$name" -d "{dest}"\n'
            f'          rm -f "{dest}/$name"\n'
            f"          ;;\n"
            f"      esac\n"
            f"    done"
        )
    return "\n".join(cmds)

def generate_github_fetch_commands():
    ds = DATASETS[CASE_STUDY]
    files = ds.get("github", {}).get("files", [])
    cmds = []
    for f in files:
        url = f["url"]
        raw_url = (
            url.replace("github.com", "raw.githubusercontent.com")
               .replace("/blob/", "/")
        )
        dest = f["dest"]
        filename = raw_url.split("/")[-1]
        cmds.append(f'mkdir -p "{dest}"')
        cmds.append(f'echo "Downloading {filename} -> {dest}"')
        cmds.append(
            f'curl -L --fail --retry 3 --retry-delay 2 '
            f'-o "{dest}/{filename}" "{raw_url}"'
        )
    return "\n".join(cmds)
