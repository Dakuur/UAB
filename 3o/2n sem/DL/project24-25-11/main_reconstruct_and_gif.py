import os
import torch
import yaml
from utils.gif_creator import ReconstructionGIFBuilder
from utils.dataloader import BEVDataset
from torch.utils.data import DataLoader

# 🟡 NOM de l’experiment que vols visualitzar
experiment_name = "test_RGB_1"

# 📄 Carregar directori base de resultats des de directories.yaml
with open("directories.yaml", "r") as f:
    dirs = yaml.safe_load(f)

results_root = dirs["results_dir"]  # ex: D:/DADES/Results_Projecte
experiment_dir = os.path.join(results_root, experiment_name)

# 📄 Buscar fitxer YAML dins la carpeta (pot ser 'params' o 'params.yaml')
possible_yaml = [f for f in os.listdir(experiment_dir) if f.startswith("params")]
if not possible_yaml:
    raise FileNotFoundError(f"No s'ha trobat cap fitxer 'params' a {experiment_dir}")
params_path = os.path.join(experiment_dir, possible_yaml[0])

# 🔁 Carregar la configuració
with open(params_path, "r") as f:
    config = yaml.safe_load(f)

# 📦 Preparar camins
checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
if not pth_files:
    raise FileNotFoundError(f"No s'ha trobat cap fitxer .pth a {checkpoint_dir}")
checkpoint_path = os.path.join(checkpoint_dir, pth_files[0])

# 🧠 Inicialitzar model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion_method = config["architecture"].get("fusion_method", "mean")
model_type = config["architecture"].get("model", "unet")

if model_type == "unet":
    from models.encoders_unet import BEVModel
elif model_type == "transformer":
    from models.encoders_unet_transformers import BEVModel
else:
    raise ValueError(f"Model type desconegut: {model_type}")

model = BEVModel(
    num_cameras=config["dataset"]["cameras"]["num_cameras"],
    in_ch=config["dataset"]["cameras"].get("in_channels", 3),
    out_ch=config["dataset"].get("out_channels", 3),
    fusion_method=fusion_method
).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# 🔄 Dataset de validació
dataset = BEVDataset(
    root_dir=dirs["dataset_dir"],
    subfolders=config["dataset"]["validation"],
    num_cameras=config["dataset"]["cameras"]["num_cameras"],
    semantic_segmentation=config["dataset"]["semantic_segmentation"]
)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 🎞️ Crear reconstruccions + GIF
reconstruction_dir = os.path.join(experiment_dir, "reconstructions")
builder = ReconstructionGIFBuilder(
    model,
    dataloader=loader,
    save_dir=reconstruction_dir,
    semantic=config["dataset"]["semantic_segmentation"],
    num_classes=30
)

builder.generate_images_with_confusion()
builder.generate_gif(os.path.join(reconstruction_dir, "result.gif"))
