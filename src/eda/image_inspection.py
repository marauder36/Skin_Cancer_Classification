# install first:
# pip install datasets pillow matplotlib pandas numpy
import os

from huggingface_hub import login

from src import HFLogin

login(HFLogin)

from datasets import load_dataset
from PIL import Image, ImageTk
import tkinter as tk
import random
import numpy as np


DATASET_NAME = "Falah/skin-cancer"


# Load dataset
dataset = load_dataset(
    DATASET_NAME,
    split="train",
    verification_mode="no_checks"
)

print(dataset)
print("Columns:", dataset.column_names)
print("Features:", dataset.features)
print("First example:", dataset[0])


def find_image_column(example):
    for key, value in example.items():
        if isinstance(value, Image.Image):
            return key

    raise ValueError("Nu am găsit coloană cu imagini.")


IMAGE_COLUMN = find_image_column(dataset[0])


def get_image_info(img):
    arr = np.array(img)

    if len(arr.shape) == 2:
        channels = 1
    else:
        channels = arr.shape[2]

    return {
        "Width": img.width,
        "Height": img.height,
        "Channels": channels,
        "Mode": img.mode,
        "Pixel min": arr.min(),
        "Pixel max": arr.max(),
        "Pixel mean": round(float(arr.mean()), 2),
        "Pixel std": round(float(arr.std()), 2),
        "Aspect ratio": round(img.width / img.height, 3),
    }


def decode_value(column_name, value):
    """
    Decodează label-uri numerice dacă datasetul are ClassLabel.
    Exemplu: 0 -> benign, 1 -> malignant
    """

    try:
        feature = dataset.features[column_name]

        if hasattr(feature, "names") and isinstance(value, int):
            return f"{value} ({feature.names[value]})"

    except Exception:
        pass

    return value


def get_metadata_text(example):
    metadata_text = "Dataset metadata / labels:\n"

    found_metadata = False

    for key, value in example.items():
        if key != IMAGE_COLUMN:
            found_metadata = True
            decoded_value = decode_value(key, value)
            metadata_text += f"{key}: {decoded_value}\n"

    if not found_metadata:
        metadata_text += "Nu există alte coloane în afară de imagine.\n"

    return metadata_text


def load_random_image():
    global current_photo

    index = random.randint(0, len(dataset) - 1)
    example = dataset[index]

    img = example[IMAGE_COLUMN]

    if not isinstance(img, Image.Image):
        img = Image.open(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    info = get_image_info(img)

    display_img = img.copy()
    display_img.thumbnail((600, 500))

    current_photo = ImageTk.PhotoImage(display_img)
    image_label.config(image=current_photo)

    text = f"Index: {index}\n\n"

    text += "Image info:\n"
    for key, value in info.items():
        text += f"{key}: {value}\n"

    text += "\n"
    text += get_metadata_text(example)

    info_label.config(text=text)


root = tk.Tk()
root.title("Skin Lesion Dataset Viewer")
root.geometry("950x750")


title_label = tk.Label(
    root,
    text="Skin Lesion Dataset Viewer",
    font=("Arial", 18, "bold")
)
title_label.pack(pady=10)


image_label = tk.Label(root)
image_label.pack(pady=10)


info_label = tk.Label(
    root,
    text="",
    font=("Arial", 11),
    justify="left",
    anchor="w"
)
info_label.pack(pady=10)


button = tk.Button(
    root,
    text="Random image",
    font=("Arial", 14),
    command=load_random_image
)
button.pack(pady=15)


current_photo = None

load_random_image()

root.mainloop()