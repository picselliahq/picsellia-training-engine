COV_ARGS=""
if [[ "$1" == "cov" ]]; then
    echo "Coverage argument specified, preparing coverage paths..."
    # Trouver tous les chemins qui contiennent conftest.py dans tests/
    IFS=$'\n' read -d '' -r -a paths_array <<< "$(find tests/ -type f -name conftest.py -exec dirname {} \;)"

    for path in "${paths_array[@]}"; do
        # Convertit le chemin de tests/ à src/
        module_path=$(echo "$path" | sed 's|tests|src|g')
        if [[ -d "$module_path" ]]; then
            for module in $(find "$module_path" -type d); do
                module=${module//\//.}
                module=${module#src.}
                COV_ARGS="$COV_ARGS --cov=src.$module"
            done
        elif [[ -f "$module_path.py" ]]; then
            # Suppose que c'est un fichier Python direct
            file_module=${module_path#src/}
            file_module=${file_module%.py}
            COV_ARGS="$COV_ARGS --cov=src.$file_module"
        fi
    done
    echo "Running pytest with coverage on: $COV_ARGS"
else
    echo "No coverage argument specified."
fi

# Exécuter pytest pour chaque répertoire trouvé
for path in "${paths_array[@]}"; do
    echo "Running pytest in $path"
    pytest $path $COV_ARGS
done
