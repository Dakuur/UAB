import os
import glob
from torchvision.transforms.functional import resize as resize_tensor
import numpy as np
import torchvision
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import yaml
import random


class BEVDataset(Dataset):
    def __init__(self, root_dir, subfolders, num_cameras=2, semantic_segmentation=False, dropout_prob=0.0, percentage=1.0, image_size=(300, 300)):
        self.root_dir = root_dir
        self.subfolders = subfolders  # lista de subcarpetas
        self.num_cameras = num_cameras
        self.image_size = tuple(image_size)
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        self.prefix = "ss_" if semantic_segmentation else ""
        self.ss = semantic_segmentation
        self.dropout_prob = dropout_prob

        self.samples = []

        for subfolder in subfolders:
            folder_path = os.path.join(root_dir, subfolder)
            dirs = [f"img{i+1}" for i in range(num_cameras)] + ["bev"]

            all_timestamps = []

            for d in dirs:
                folder_name = f"{self.prefix}{d}"  # ex: ss_img1, ss_bev
                full_path = os.path.join(folder_path, folder_name)

                if not os.path.exists(full_path):
                    all_timestamps.append(set())
                    continue

                # debug
                """print(f"Processing folder: {full_path}")
                print(f"Folder name: {folder_name}")
                exit(1)"""

                if self.ss:
                    pattern = os.path.join(full_path, f"{folder_name}_*.semantic.npy")
                else:
                    pattern = os.path.join(full_path, f"{folder_name}_*.png")
                files = glob.glob(pattern)
                timestamps = set(
                    os.path.basename(f)
                    .replace(f"{folder_name}_", "")
                    .replace(".png", "")
                    .replace(".semantic.npy", "")
                    for f in files
                )
                if percentage < 1:
                    timestamps = set(list(timestamps)[:int(len(timestamps) * percentage)])

                if percentage < 1.0:
                    timestamps = set(list(timestamps)[:int(len(timestamps) * percentage):])
                all_timestamps.append(timestamps)
            if all_timestamps:
                common_ts = set.intersection(*all_timestamps)
                for ts in common_ts:
                    self.samples.append((subfolder, ts))

        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subfolder, ts = self.samples[idx]
        folder_path = os.path.join(self.root_dir, subfolder)
        inputs = []

        for cam_id in range(1, self.num_cameras + 1):
            simulate_dropout = random.random() < self.dropout_prob
            img_folder = f"{self.prefix}img{cam_id}"
            file_base = f"{self.prefix}img{cam_id}_{ts}"
            cam_path = os.path.join(folder_path, img_folder)

            if simulate_dropout:
                img = Image.new("RGB", self.image_size, (0, 0, 0))  # Imatge negra amb la mida adequada
                img = self.transform(img)
            else:
                if self.ss:
                    npy_path = os.path.join(cam_path, file_base + ".semantic.npy")
                    ss_img = np.load(npy_path)
                    one_hot = np.eye(30)[ss_img]  # (H, W, 30)
                    one_hot = torch.tensor(one_hot).permute(2, 0, 1).float()  # (30, H, W)
                    one_hot = resize_tensor(one_hot, self.image_size)  # resize per tensor
                    img = one_hot
                else:
                    img_path = os.path.join(cam_path, file_base + ".png")
                    img = Image.open(img_path).convert("RGB")
                    img = self.transform(img)

            inputs.append(img)

        # BEV
        bev_folder = f"{self.prefix}bev"
        bev_base = f"{self.prefix}bev_{ts}"
        bev_path = os.path.join(folder_path, bev_folder)

        if self.ss:
            npy_path = os.path.join(bev_path, bev_base + ".semantic.npy")
            ss_bev = np.load(npy_path)
            one_hot = np.eye(30)[ss_bev]  # (H, W, 30)
            one_hot = torch.tensor(one_hot).permute(2, 0, 1).float()
            one_hot = resize_tensor(one_hot, self.image_size)  # resize
            output = one_hot
        else:
            img_path = os.path.join(bev_path, bev_base + ".png")
            bev_img = Image.open(img_path).convert("RGB")
            output = self.transform(bev_img)

        input_tensor = torch.stack(inputs, dim=0)
        return input_tensor, output


if __name__ == "__main__":
    # Ejemplo de uso
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("directories.yaml", "r") as f:
        data_dir = yaml.safe_load(f)["dataset_dir"]

    print(f"Torch avaliable: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    data_dir = data_dir
    cams = config["dataset"]["cameras"]["num_cameras"]
    ss = config["dataset"]["semantic_segmentation"]

    print(f"Loading dataset from: {data_dir}")

    dataset = BEVDataset(root_dir=data_dir, num_cameras=cams, semantic_segmentation=ss, subfolders=config["dataset"]["training"])
    #dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    dataloader = DataLoader(dataset, batch_size=config["dataset"]["batch_size"], shuffle=True)

    print(f"Total samples: {len(dataset)}")

    for image, label in dataloader:
        print(f"Input shape: {image.shape},\nBEV shape: {label.shape}")
        break  # Solo para verificar la forma de los datos, eliminar en producción

    for image, label in dataloader:
        # Crear un grid 3x3 con las 8 cámaras y el BEV
        grid_images = torch.cat([image, label.unsqueeze(1)], dim=1)  # Añadir BEV como la novena imagen
        grid_images = grid_images.view(-1, 3, 300, 300)  # Asegurar que cada imagen tiene el formato correcto
        grid = torchvision.utils.make_grid(grid_images[:9], nrow=3)  # Crear el grid 3x3

        # Guardar el grid como imagen PNG
        output_path = os.path.join(data_dir, "grid_output.png")
        torchvision.utils.save_image(grid, output_path)
        print(f"Grid guardado en: {output_path}")
          # Solo para verificar, eliminar en producción
