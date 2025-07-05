import os
import shutil
from pathlib import Path
import re

def copiar_params(base_dir, destino_dir):
    base_dir = Path(base_dir).resolve()
    destino_dir = Path(destino_dir).resolve()
    destino_dir.mkdir(parents=True, exist_ok=True)

    semseg_re = re.compile(r"semseg-(True|False)")

    for root, _, files in os.walk(base_dir):
        if "params.yaml" in files:
            ruta_origen = Path(root) / "params.yaml"
            nombre_carpeta = Path(root).name

            # Buscar semseg en el nombre de la carpeta
            match = semseg_re.search(nombre_carpeta)
            if match:
                semseg_val = match.group(1)
                subfolder = "ss" if semseg_val == "True" else "rgb"
            else:
                subfolder = "rgb"  # Por defecto

            destino_subdir = destino_dir / subfolder
            destino_subdir.mkdir(parents=True, exist_ok=True)
            ruta_destino = destino_subdir / f"{nombre_carpeta}.yaml"

            if ruta_destino.exists():
                raise FileExistsError(f"Archivo duplicado: {ruta_destino}")

            shutil.copy2(ruta_origen, ruta_destino)
            print(f"Copiado: {ruta_origen} -> {ruta_destino}")

copiar_params(
    "/data1/121-1/Experiments/dmorillo/deeplearning/results/grid_search",
    "/data1/121-1/Experiments/dmorillo/deeplearning/results/grid_search_all_configs"
)
