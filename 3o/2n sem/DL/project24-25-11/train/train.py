import os
import torch
import yaml
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from train.component_selector import get_loss_components, get_scheduler
from train.training_utils import save_checkpoint, log_images
from utils.dataloader import BEVDataset
from time import time 
#from sklearn.metrics import accuracy_score, precision_score, f1_score


def train(config, dirs):

    start_time = time()

    # Paths
    data_dir = dirs["dataset_dir"]
    results_dir = dirs["results_dir"]


    experiment_path = os.path.join(results_dir, config["experiment"])
    checkpoint_path = os.path.join(experiment_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    output_config = os.path.join(experiment_path, "params.yaml")
    with open(output_config, "w") as out:
        yaml.dump(config, out)

    # Dataset
    train_folders = config["dataset"]["training"]
    val_folders = config["dataset"]["validation"]
    cams = config["dataset"]["cameras"]["num_cameras"]
    ss = config["dataset"]["semantic_segmentation"]
    batch_size = config["dataset"]["batch_size"]
    dropout = config["dataset"]["cameras"]["dropout_prob"]
    train_percentage = config["dataset"]["train_percentage"]
    val_percentage = config["dataset"]["val_percentage"]
    num_workers=config["dataset"]["num_workers"]

    model_type = config["architecture"].get("model", "unet")
    if model_type == "unet":
        from models.encoders_unet import BEVModel
    elif model_type == "transformer":
        from models.encoders_unet_transformers import BEVModel

    print(f"Loading training dataset from: {train_folders}")

    train_dataset = BEVDataset(root_dir=data_dir, subfolders=train_folders, num_cameras=cams, semantic_segmentation=ss, dropout_prob=dropout, percentage=train_percentage, image_size=config["dataset"]["cameras"]["image_size"])
    if len(train_dataset) == 0:
        raise ValueError("No training data found. Please check the dataset path and folders.")

    print(f"Loading validation dataset from: {val_folders}")
    val_dataset = BEVDataset(root_dir=data_dir, subfolders=val_folders, num_cameras=cams, semantic_segmentation=ss, dropout_prob=dropout, percentage=val_percentage, image_size=config["dataset"]["cameras"]["image_size"])
    if len(val_dataset) == 0:
        raise ValueError("No validation data found. Please check the dataset path and folders.")

    if num_workers > 1:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("")
    print(f"Semantic segmentation: {ss}")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Input shape: {next(iter(train_loader))[0].shape} -> [B, C, D, W, H]")
    print(f"Target shape: {next(iter(train_loader))[1].shape} -> [B, D, W, H]")
    print("")

    #print(f"No errors found in dataset: {len(dataset)}")
    #exit(1)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion = config["architecture"].get("fusion_method", "mean")

    channels = next(iter(train_loader))[0].shape[2]

    model = BEVModel(num_cameras=cams, fusion_method=fusion, in_ch=channels, out_ch=channels).to(device).float()


    criterion = get_loss_components(config["training"]["loss_components"], ss_mode=ss)


    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"]["initial"])
    lr_config = config["training"]["learning_rate"]
    scheduler_type = lr_config.get("type", "step")
    scheduler = get_scheduler(optimizer, lr_config, scheduler_type)

    # WandB
    wandb.init(project="BEV-Reconstruction", config=config, name=config["experiment"], reinit=True)
    wandb.watch(model, log="all")

    # Training loop
    num_epochs = config["training"]["epochs"]
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for inputs, targets in tqdm(train_loader, total=len(train_loader)):

            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # Validation
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device).float(), targets.to(device).float()
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        wandb_log = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        }

        wandb.log(wandb_log)

        if (epoch + 1) % config["training"]["save_interval"] == 0:
            save_checkpoint(model, checkpoint_path, epoch)

        if (epoch + 1) % config["training"]["save_interval"] == 0:
            log_images(model, val_loader, device, epoch, ss)

        if scheduler_type == "plateau":
            scheduler.step(avg_val_loss)
        elif scheduler is not None:
            scheduler.step()

    log_images(model, val_loader, device, epoch, ss)

    # Final save
    final_checkpoint_file = os.path.join(checkpoint_path, "model_final.pth")
    torch.save(model.state_dict(), final_checkpoint_file)

    end_time = time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Extract the final validation loss and total time to result yaml
    config["total_time"] = end_time - start_time
    config["final_validation_loss"] = avg_val_loss
    with open(output_config, "w") as out:
        yaml.dump(config, out)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("../params.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("../directories.yaml", "r") as f:
        dirs = yaml.safe_load(f)

    train(config, dirs)