import os
import torch
import wandb
from utils.semantic_map import ss2rgb

def save_checkpoint(model, path, epoch):
    checkpoint_file = os.path.join(path, f"model_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), checkpoint_file)
    print(f"Checkpoint saved to {checkpoint_file}")
    wandb.save(checkpoint_file)

def log_images(model, val_loader, device, epoch, ss):
    model.eval()
    inputs, targets = next(iter(val_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)

    pred_img = outputs[0].detach().cpu().permute(1, 2, 0).numpy()
    target_img = targets[0].detach().cpu().permute(1, 2, 0).numpy()

    if ss:
        pred_img = ss2rgb(pred_img)
        target_img = ss2rgb(target_img)
    else:
        pred_img = (pred_img * 255).clip(0, 255).astype("uint8")
        target_img = (target_img * 255).clip(0, 255).astype("uint8")


    wandb.log({
        "prediction": wandb.Image(pred_img, caption="Prediction Image"),
        "ground_truth": wandb.Image(target_img, caption="Ground Truth Image"),
        "epoch": epoch + 1
    })