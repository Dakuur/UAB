import torch
import yaml
from train.train import train
import glob
import os
import numpy as np

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    device_gpu = torch.cuda.current_device()
    print(f"Using GPU: {torch.cuda.get_device_name(device_gpu)}, index: {device_gpu}")

    dirs = yaml.safe_load(open("directories.yaml"))
    config = yaml.safe_load(open("params.yaml"))

    train(config, dirs)



"""fusion_methods = ["mean", "max", "min", "concat"]

for fusion_method in fusion_methods:
    config["architecture"]["fusion_method"] = fusion_method

    for lr in [0.005, 0.001]:
        config["training"]["learning_rate"]["initial"] = lr
        
        for ss in [True, False]:
            
            config["dataset"]["semantic_segmentation"] = ss
            if ss:
                config["training"]["loss"] = "CrossEntropy"
            else:
                config["training"]["loss"] = "MSE"

            config["experiment"] = f"ss{ss}_{fusion_method}_lr-{lr:.4f}" # Nom unic de l'experiment
            train(config, dirs)"""



"""
Multi-gpu:
import torch
import yaml
import os
import numpy as np
import argparse
import multiprocessing as mp
from queue import Queue
from train.train import train

def run_queue(gpu_id, experiment_queue, dirs):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    while not experiment_queue.empty():
        config = experiment_queue.get()
        print(f"[GPU {gpu_id}] Starting {config['experiment']}")
        train(config, dirs)
        print(f"[GPU {gpu_id}] Finished {config['experiment']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", required=True,
                        help="Lista de GPUs disponibles (por índice), por ejemplo: --gpus 0 1 2 3")
    args = parser.parse_args()
    gpus = args.gpus

    dirs = yaml.safe_load(open("directories.yaml"))

    fusion_methods = ["mean", "max", "min", "concat"]
    lrs = list(np.arange(0.0001, 0.001, 0.0001))

    # Crear cola de experimentos por GPU
    queues = [mp.Queue() for _ in gpus]
    total_experiments = 0
    for i, fusion_method in enumerate(fusion_methods):
        for lr in lrs:
            config = yaml.safe_load(open("params.yaml"))
            config["architecture"]["fusion_method"] = fusion_method
            config["training"]["learning_rate"]["initial"] = lr
            config["experiment"] = f"exp_{fusion_method}_lr-{lr:.4f}"
            gpu_index = total_experiments % len(gpus)
            queues[gpu_index].put(config)
            total_experiments += 1

    mp.set_start_method("spawn")
    processes = []
    for gpu_id, q in zip(gpus, queues):
        p = mp.Process(target=run_queue, args=(gpu_id, q, dirs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

"""

"""
yaml_files = glob.glob(os.path.join("david_experiments", "*.yaml"))
# Alla tinc .yaml(s) diferents per cada experiment
# He posat la carpeta david_experiments/ al .gitignore btw
print(f"Experiment files: {yaml_files}")

for yaml_file in yaml_files:
    config = yaml.safe_load(open(yaml_file))

    train(config, dirs) # Entrenar amb la configuració de cada experiment
"""
