import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = "smile_model.h5"
IMG_SIZE   = 224
THRESHOLD  = 0.4   

print("Loading Smiley model...")
model = load_model(MODEL_PATH)
print("Model ready!\n")

def predict_image(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = model.predict(arr, verbose=0)[0][0]

    if prob >= THRESHOLD:
        label      = "Smiling 😊"
        confidence = prob * 100
        color      = "#27ae60"  
    else:
        label      = "Not Smiling 😐"
        confidence = (1 - prob) * 100
        color      = "#e74c3c"   

    return label, confidence, color, img

def predict_single(image_path):
    label, confidence, color, img = predict_image(image_path)

    print(f"Image     : {os.path.basename(image_path)}")
    print(f"Result    : {label}")
    print(f"Confidence: {confidence:.1f}%")

    plt.figure(figsize=(5, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{label}\nConfidence: {confidence:.1f}%",
              fontsize=14, fontweight='bold', color=color, pad=15)
    plt.tight_layout()
    plt.show()

def predict_folder(folder_path):
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith(extensions)]

    if not images:
        print(f"No images found in: {folder_path}")
        return

    print(f"Found {len(images)} images. Classifying...\n")

    results = []
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        label, confidence, color, _ = predict_image(img_path)
        results.append((img_name, label, confidence, color))
        print(f"  {img_name:30s} → {label}  ({confidence:.1f}%)")

    n = min(len(results), 12)  
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (img_name, label, confidence, color) in enumerate(results[:n]):
        img_path = os.path.join(folder_path, img_name)
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"{label}\n{confidence:.1f}%",
                          fontsize=10, color=color, fontweight='bold')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    smiling     = sum(1 for _, l, _, _ in results if "Smiling 😊" in l)
    not_smiling = len(results) - smiling

    fig.suptitle(f"Smiley Results — {smiling} Smiling / {not_smiling} Not Smiling",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('batch_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nResults grid saved to: batch_predictions.png")
    print(f"\nSummary: {smiling} Smiling | {not_smiling} Not Smiling")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smiley - Smile Detection")
    parser.add_argument("--image",  type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    args = parser.parse_args()

    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: File not found: {args.image}")
            sys.exit(1)
        predict_single(args.image)

    elif args.folder:
        if not os.path.isdir(args.folder):
            print(f"Error: Folder not found: {args.folder}")
            sys.exit(1)
        predict_folder(args.folder)

    else:
        print("Usage examples:")
        print("  python predict.py --image  photo.jpg")
        print("  python predict.py --folder test_folder/smiling/")