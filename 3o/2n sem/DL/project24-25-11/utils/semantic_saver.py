import os
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


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
    29: (180, 165, 180),
}

CITYSCAPES_ID = {CITYSCAPES_SS[key]: key for key in CITYSCAPES_SS.keys()}

def rgb2ss(rgb_img):
    rgb_img = np.asarray(rgb_img, dtype=np.int64)
    w, h, _ = rgb_img.shape

    ss_img = np.zeros((w, h), dtype=np.int8)

    for u in range(w):
        for v in range(h):
            rgb = tuple(rgb_img[u, v, :])
            ss_img[u, v] = CITYSCAPES_ID.get(rgb, 0)  # 0 si color desconocido

    return ss_img
    
def process_directory(root_dir):
    for session_folder in os.listdir(root_dir):
        session_path = os.path.join(root_dir, session_folder)

        if not os.path.isdir(session_path):
            continue

        # Buscar subcarpetas que empiecen por 'ss_'
        for subfolder in os.listdir(session_path):
            if not subfolder.startswith("ss_"):
                continue

            ss_folder_path = os.path.join(session_path, subfolder)
            if not os.path.isdir(ss_folder_path):
                continue

            # Contar archivos
            png_files = [f for f in os.listdir(ss_folder_path) if f.endswith(".png")]
            npy_files = [f for f in os.listdir(ss_folder_path) if f.endswith(".semantic.npy")]

            if len(png_files) == len(npy_files):
                print(f"Saltando {ss_folder_path} (ya procesado)")
                continue

            for file in tqdm(png_files, desc=f"Procesando {subfolder}", unit="img"):
                file_path = os.path.join(ss_folder_path, file)
                try:
                    img = Image.open(file_path).convert("RGB")
                    img_np = np.array(img)

                    ss_map = rgb2ss(img_np)

                    output_path = os.path.splitext(file_path)[0] + ".semantic.npy"
                    np.save(output_path, ss_map)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


process_directory("D:/datasets")
"""img = np.load("D:/datasets/hola/rosbag2_2025_05_08-12_07_12/ss_img1/ss_img1_1746698837_745686769.semantic.npy")
print(img)"""