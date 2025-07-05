import torch
import torch.nn as nn
from train.MSE_cannyloss import MSECannyLoss

def get_scheduler(optimizer, lr_config, scheduler_type="step"):
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_config.get("step_size", 25),
            gamma=lr_config.get("gamma", 0.5)
        )

    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=lr_config.get("gamma", 0.5),
            patience=lr_config.get("patience", 5),
        )

    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=lr_config.get("T_max", 50)
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}") #T'avisa que el lr esta mal


def get_classes_weights(device=None):
    """
    Retorna un tensor amb els pesos per a cada classe,
    donant més importància a les classes rellevants per a la conducció i la carretera.

    Args:
        device (torch.device o None): dispositiu on es vol col·locar el tensor (cpu o cuda).
    
    Returns:
        torch.Tensor: tensor 1D amb els pesos de cada classe.
    """
    num_classes = 30
    weights = torch.ones(num_classes)  # Inicialitza els pesos a 1 per a totes les classes

    # Classes que considerem importants i els seus pesos associats
    important_classes = {
        1: 5.0,   # Carreteres
        2: 5.0,   # Voreres (SideWalks)
        14: 5.0,  # Cotxes
        15: 5.0,  # Camions
        16: 5.0,  # Autobusos
        12: 5.0,  # Vianants
        13: 5.0,  # Conductors (Riders)
        24: 5.0,  # Marcatges de la carretera (RoadLine)
    }

    print("Pesos de les classes:")
    for cls_idx, w in important_classes.items():
        print(f"Classe {cls_idx}: {w}")

    # Assigna el pes corresponent a cada classe important
    for cls_idx, w in important_classes.items():
        weights[cls_idx] = w

    # Mou el tensor al dispositiu indicat, si s'ha especificat
    if device:
        weights = weights.to(device)

    return weights


def get_loss_components(loss_config: dict, ss_mode: bool):
    from utils.losses import DiceLoss, SSIMLoss, PSNRLoss
    import lpips
    import torch.nn as nn

    losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

    for name, weight in loss_config.items():
        if name == "MSE":
            losses.append((nn.MSELoss(), weight))
        elif name == "LPIPS" and not ss_mode:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            losses.append((lpips.LPIPS(net='alex').to(device), weight))

        elif name == "PSNR" and not ss_mode:
            losses.append((PSNRLoss(), weight))
        elif name == "SSIM" and not ss_mode:
            losses.append((SSIMLoss(), weight))
        elif name == "Dice" and ss_mode:
            losses.append((DiceLoss(), weight))
        elif name == "CrossEntropy" and ss_mode:
            class_weights = get_classes_weights(device=device)
            #torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
            losses.append((nn.CrossEntropyLoss(weight=class_weights), weight))
        

    def combined_loss(pred, target):
        total = 0.0
        for loss_fn, w in losses:
            loss_value = loss_fn(pred, target)
            if loss_value.dim() != 0:
                loss_value = loss_value.mean()
            total += w * loss_value

        return total

    return combined_loss

"""
def test_get_loss_components_crossentropy():
    # Configuración de pérdidas: solo CrossEntropy con peso 1.0
    loss_config = {"CrossEntropy": 1.0}
    ss_mode = True  # Para activar el modo segmentación semántica

    # Obtenemos la función de pérdida combinada
    combined_loss = get_loss_components(loss_config, ss_mode)

    # Creamos un batch dummy con 2 ejemplos, 30 clases, tamaño 4x4 (por ejemplo)
    batch_size = 2
    num_classes = 30
    height, width = 300,300

    # Predicciones random (logits sin softmax) - shape (N, C, H, W)
    preds = torch.randn(batch_size, num_classes, height, width, requires_grad=True)

    # Targets dummy con índices de clase válidos entre 0 y 29 - shape (N, H, W)
    targets = torch.randint(0, num_classes, (batch_size, height, width), dtype=torch.long)

    # Calculamos la pérdida
    loss_val = combined_loss(preds, targets)

    print("Pérdida CrossEntropy calculada:", loss_val.item())

    # Comprobamos que la pérdida es un escalar y tiene gradientes
    assert loss_val.dim() == 0, "La pérdida debe ser un escalar."
    loss_val.backward()  # Esto debería funcionar sin errores
    print("Backward OK, gradientes calculados.")

if __name__ == "__main__":
    test_get_loss_components_crossentropy()
"""