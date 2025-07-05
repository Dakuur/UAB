import torch
import matplotlib.pyplot as plt
import seaborn as sns

def compute_confusion_matrix(output, targets, num_classes):
    
    """
    Retorna la matriu de confusió a partir de les prediccions i les etiquetes reals.
    Args:   
        output (Tensor): Prediccions del model (batch_size x num_classes x H x W).
        targets (Tensor): Etiquetes reals (batch_size x num_classes x H x W).
        num_classes (int): Nombre de classes.
    Returns:
        Tensor: Matriu de confusió (num_classes x num_classes).
    """
    # Convertim les prediccions (one-hot) a índexs de classes (1D)
    output = torch.argmax(output, dim=1).view(-1)
    targets = torch.argmax(targets, dim=1).view(-1)

    # Creem una màscara per filtrar valors vàlids de les etiquetes
    mask = (targets >= 0) & (targets < num_classes)
    output = output[mask]
    targets = targets[mask]

    # Inicialitzem la matriu de confusió (num_classes x num_classes)
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    # Per cada parell (etiqueta real, predicció), augmentem el comptador corresponent
    for t, p in zip(targets, output):
        conf_matrix[t, p] += 1

    # Retornem la matriu de confusió
    return conf_matrix


def plot_confusion_matrix(conf_matrix, class_names=None, normalize=False, title="Matriu de confusió"):
    """
    Mostra la matriu de confusió amb valors absoluts o normalitzats.

    Args:
        conf_matrix (Tensor): Matriu de confusió (num_classes x num_classes).
        class_names (list): Noms de les classes (opcional).
        normalize (bool): Si True, normalitza per files (recall per classe).
        title (str): Títol del gràfic.
    """
    cm = conf_matrix.float()
    if normalize:
        cm = cm / cm.sum(1, keepdim=True).clamp(min=1e-8)

    cm_np = cm.cpu().numpy()

    plt.figure(figsize=(10, 8))
    fmt = ".2f" if normalize else ".0f"
    sns.heatmap(cm_np, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names if class_names else range(conf_matrix.shape[0]),
                yticklabels=class_names if class_names else range(conf_matrix.shape[0]))
    plt.xlabel("Predicció")
    plt.ylabel("Etiqueta real")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


def compute_metrics_from_conf_matrix(conf_matrix):
    # True Positives: elements de la diagonal principal
    TP = torch.diag(conf_matrix)
    # False Positives: suma per columna menys TP
    FP = conf_matrix.sum(0) - TP
    # False Negatives: suma per fila menys TP
    FN = conf_matrix.sum(1) - TP
    # True Negatives: suma total menys TP, FP i FN
    TN = conf_matrix.sum() - (TP + FP + FN)
    # Precision per classe = TP / (TP + FP) amb petit epsilon per evitar divisió per zero
    precision = TP / (TP + FP + 1e-8)
    # Recall per classe = TP / (TP + FN)
    recall = TP / (TP + FN + 1e-8)
    # F1 score per classe = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    # Accuracy global = suma TP / total elements
    accuracy = TP.sum().float() / conf_matrix.sum().float()

    # Guardem totes les mètriques en un diccionari per retornar
    metrics = {
        "accuracy": accuracy.item(),
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "mean_precision": precision.mean().item(),
        "mean_recall": recall.mean().item(),
        "mean_f1": f1.mean().item(),
    }

    return metrics

if __name__ == "__main__":
    # Exemple d'ús
    num_classes = 3
    output = torch.randn(2, num_classes, 4, 4)  # Simulació de prediccions
    targets = torch.randint(0, num_classes, (2, num_classes, 4, 4))  # Simulació d'etiquetes reals

    conf_matrix = compute_confusion_matrix(output, targets, num_classes)
    plot_confusion_matrix(conf_matrix, class_names=["Fons", "Classe1", "Classe2"], normalize=True)
    metrics = compute_metrics_from_conf_matrix(conf_matrix)
    for k, valors in metrics.items():
        if isinstance(valors, torch.Tensor):
            print(f"{k}: {valors.tolist()}")
        else:
            print(f"{k}: {valors}")

    plot_confusion_matrix(conf_matrix, class_names=["Fons", "Classe1", "Classe2"], normalize=True)