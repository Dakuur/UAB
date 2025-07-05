import os
import sys
import torch
import yaml
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Afegim el path base per poder importar altres mòduls
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.encoders_unet import BEVModel
from dataloader import BEVDataset
from utils.semantic_map import ss2rgb  # Per semantic segmentation


# Funció per convertir una imatge RGB semantic a índexs de classe
def rgb2ss(rgb_img):
    CITYSCAPES_SS = {
        0: (0, 0, 0),
        1: (128, 64, 128),
        2: (244, 35, 232),
        3: (70, 70, 70),
        4: (102, 102, 156),
        5: (190, 153, 153),
        6: (153, 153, 153),
        7: (250, 170, 30),
        8: (220, 220, 0),
        9: (107, 142, 35),
        10: (152, 251, 152),
        11: (70, 130, 180),
        12: (220, 20, 60),
        13: (255, 0, 0),
        14: (0, 0, 142),
        15: (0, 0, 70),
        16: (0, 60, 100),
        17: (0, 80, 100),
        18: (0, 0, 230),
        19: (119, 11, 32),
        21: (110, 190, 160),
        22: (170, 120, 50),
        23: (55, 90, 80),
        24: (45, 60, 150),
        25: (157, 234, 50),
        26: (81, 0, 81),
        27: (150, 100, 100),
        28: (230, 150, 140),
        29: (180, 165, 180)
    }
    color_to_class = {v: k for k, v in CITYSCAPES_SS.items()}
    h, w, _ = rgb_img.shape
    flat_rgb = rgb_img.reshape(-1, 3)
    output = np.array([
        color_to_class.get(tuple(pixel), 255)  # 255 = ignore
        for pixel in flat_rgb
    ])
    return output.reshape(h, w)

# Funció segura per matriu de confusió
def compute_confusion_matrix(preds, targets, num_classes):
    mask = (targets >= 0) & (targets < num_classes) & (preds != 255)
    preds = preds[mask]
    targets = targets[mask]
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        if p < num_classes:
            conf_matrix[t, p] += 1
    return conf_matrix

def main(experiment, test_map):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths bàsics
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(base_path, "directories.yaml"), "r") as f:
        dirs = yaml.safe_load(f)

    experiment_name = experiment
    results_root = dirs["results_dir"]
    experiment_dir = os.path.join(results_root, experiment_name)
    params_path = os.path.join(experiment_dir, "params.yaml")
    checkpoint_path = os.path.join(experiment_dir, "checkpoints", "model_final.pth")
    reconstruction_dir = os.path.join(experiment_dir, "reconstructions")
    #os.makedirs(reconstruction_dir, exist_ok=True)

    # Configuració de l'experiment
    with open(params_path, "r") as f:
        config = yaml.safe_load(f)

    # Dataset
    data_dir = dirs["dataset_dir"]
    test_data = test_map

    cams = config["dataset"]["cameras"]["num_cameras"]
    ss = config["dataset"]["semantic_segmentation"]

    if ss:
        num_classes = 30
    else:
        num_classes = 3
    batch_size = 1

    #print("Subfolders:", subfolders)
    dataset = BEVDataset(
        root_dir=data_dir,
        num_cameras=cams,
        semantic_segmentation=ss,
        subfolders=[test_data],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Number of images in dataset: {len(dataset)}")

    # Model
    in_ch = num_classes if ss else 3
    out_ch = num_classes if ss else 3
    fusion_method = config["architecture"].get("fusion_method", "mean")

    model = BEVModel(
        num_cameras=cams,
        in_ch=in_ch,
        out_ch=out_ch,
        fusion_method=fusion_method
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    #exit(1)
    # Reconstrucció
    images = []
    #torch.set_float32_matmul_precision("high")
    with torch.no_grad():
        #with torch.amp.autocast(device_type="cuda"):
            for i, (inputs, targets) in enumerate(loader):
                if i < 0:
                    continue

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                outputs_cpu = outputs.cpu()
                targets_cpu = targets.cpu()

                del inputs, targets, outputs
                torch.cuda.empty_cache()

                for b in range(outputs_cpu.shape[0]):
                    pred = outputs_cpu[b]
                    gt = targets_cpu[b]
                    #save_path = os.path.join(reconstruction_dir, f"reconstruction_{i}_{b}.png")

                    if ss:
                        # Semantic: convertir de one-hot a RGB
                        pred_img = ss2rgb(pred.permute(1, 2, 0).numpy())
                        gt_img = ss2rgb(gt.permute(1, 2, 0).numpy())
                        
                        conf_matrix = compute_confusion_matrix(
                            pred.argmax(dim=0),
                            gt.argmax(dim=0),
                            num_classes=num_classes
                        )

                        """print(f"Max values in pred: {pred.max()} min: {pred.min()}")
                        print(f"Nan values in pred: {torch.isnan(pred).any()}")
                        exit(1)"""
                        #print(f"Max values in gt: {gt.max()} min: {gt.min()}")

                        conf_matrix_normalized = conf_matrix.numpy() / conf_matrix.numpy().sum(axis=1, keepdims=True)
                        #conf_matrix_normalized = conf_matrix.numpy()

                        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)  # 3 inches * 100 dpi = 300px
                        sns.heatmap(
                            conf_matrix_normalized,
                            annot=False,  # No mostrar números
                            fmt="d",
                            cmap="Blues",
                            cbar=False,
                            xticklabels=range(num_classes),
                            yticklabels=range(num_classes),
                            ax=ax
                        )
                        fig.tight_layout(pad=0)
                        canvas = FigureCanvas(fig)
                        canvas.draw()
                        heatmap_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                        heatmap_img = heatmap_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        plt.close(fig)

                        # Resize heatmap to match pred_img/gt_img height if needed
                        if heatmap_img.shape[0] != pred_img.shape[0]:
                            from PIL import Image as PILImage
                            heatmap_img = np.array(PILImage.fromarray(heatmap_img).resize((heatmap_img.shape[1], pred_img.shape[0])))

                        comparison = np.concatenate([pred_img, gt_img, heatmap_img], axis=1)
                        
                    else:
                        pred_img = pred.clamp(0, 1).permute(1, 2, 0).numpy()
                        pred_img = (pred_img * 255).astype(np.uint8)

                        gt_img = gt.clamp(0, 1).permute(1, 2, 0).numpy()
                        gt_img = (gt_img * 255).astype(np.uint8)

                        comparison = np.concatenate([pred_img, gt_img], axis=1)
                    images.append(comparison)

                print(i)
                if i == 200 or i == len(dataset): # Limitar a 200 imatges
                    break

    # imgs to gif
    images = [Image.fromarray(img) for img in images]

    # Guardar como GIF
    output_path = os.path.join(experiment_dir, "reconstruction.gif")
    duration= 50  # Duración de cada fotograma en milisegundos
    loop = 0  # Número de bucles (0 significa infinito)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        disposal=2
    )

    print(f"GIF guardado en: {output_path}")

if __name__ == "__main__":

    experiments = ["ss-extended"]
    # test_map = "rosbag2_2025_05_12-11_09_28_wetnoon_town02"
    test_map = "rosbag2_2025_05_12-07_06_54_clearnoon_town01"
    for e in experiments:
        main(e, test_map)