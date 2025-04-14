import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

IMAGE_DIR = "asset/IMG"
MASK_DIR = "asset/IMG/result"

custom_colormap = {
    "Background": (0, 0, 0),
    "Hat": (0, 255, 255),
    "Hair": (0, 165, 255),
    "Sunglasses": (255, 0, 255),
    "Upper-clothes": (0, 0, 255),
    "Skirt": (255, 255, 0),
    "Pants": (0, 255, 0),
    "Dress": (255, 0, 0),
    "Belt": (128, 0, 128),
    "Left-shoe": (0, 255, 255),
    "Right-shoe": (255, 140, 0),
    "Face": (200, 180, 140),
    "Left-leg": (200, 180, 140),
    "Right-leg": (200, 180, 140),
    "Left-arm": (200, 180, 140),
    "Right-arm": (200, 180, 140),
    "Bag": (0, 128, 255),
    "Scarf": (255, 20, 147)
}

def get_mask_files(image_name):
    name, _ = os.path.splitext(image_name)
    return [f for f in os.listdir(MASK_DIR) if f.startswith(name)]


def fuse_masks(mask_paths):
    final_mask = None
    label_map = {}

    for path in mask_paths:
        mask = np.array(Image.open(os.path.join(MASK_DIR, path)).convert("L"))
        label = path.split("_")[-1].replace(".png", "")

        if label not in custom_colormap:
            print(f"[!] Label ignoré : {label}")
            continue

        label_map[label] = mask

        if final_mask is None:
            final_mask = np.zeros_like(mask)

        final_mask[label_map[label] > 0] = list(custom_colormap.keys()).index(label) + 1

    return final_mask, label_map

def colorize_mask(mask, colormap_keys):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, label in enumerate(colormap_keys):
        colored[mask == (idx + 1)] = custom_colormap[label]
    return colored

def show_visualisation(image_name):
    image_path = os.path.join(IMAGE_DIR, image_name)
    mask_files = get_mask_files(image_name)

    if not mask_files:
        print(f"Aucun masque pour {image_name}")
        return

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask, label_map = fuse_masks(mask_files)
    color_mask = colorize_mask(mask, label_map.keys())

    overlay = cv2.addWeighted(image_rgb, 0.7, color_mask, 0.3, 0)


    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Image originale")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(color_mask)
    plt.title("Masque colorisé")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.show()
    plt.close()

image_list = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")])
for img in image_list:
    show_visualisation(img)
