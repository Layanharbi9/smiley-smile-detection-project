import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
import seaborn as sns
import time

TEST_DIR        = "test_folder"
MODEL_PATH      = "smile_model.h5"
IMG_SIZE        = 224
BATCH_SIZE      = 32

print("=" * 60)
print("  Smiley - Model Evaluation")
print("=" * 60)

print("\n[1] Loading model...")
model = load_model(MODEL_PATH)
print(f"   Model loaded from: {MODEL_PATH}")

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

class_names = list(test_generator.class_indices.keys())
print(f"   Classes: {test_generator.class_indices}")
print(f"   Test images: {test_generator.samples}")

print("\n[2] Running predictions...")
start_time = time.time()
y_pred_prob = model.predict(test_generator, verbose=1)
total_time  = time.time() - start_time

avg_inference_ms = (total_time / test_generator.samples) * 1000

y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_generator.classes

acc       = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)

print("\n" + "=" * 60)
print("  EVALUATION RESULTS")
print("=" * 60)
print(f"  Accuracy        : {acc * 100:.2f}%")
print(f"  Precision       : {precision * 100:.2f}%")
print(f"  Recall          : {recall * 100:.2f}%")
print(f"  F1-Score        : {f1 * 100:.2f}%")
print(f"  Avg Inference   : {avg_inference_ms:.2f} ms/image")
print("=" * 60)

print("\n  Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=1, linecolor='gray')
plt.title('Confusion Matrix - Smile Detection', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("   Confusion matrix saved to: confusion_matrix.png")

metrics_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [acc, precision, recall, f1]
colors = ['#4C9BE8', '#5DBB8A', '#F4A261', '#E76F51']

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics_names, [v * 100 for v in metrics_values],
               color=colors, width=0.5, edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f'{val * 100:.1f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.ylim(0, 115)
plt.title('Evaluation Metrics - Smiley System', fontsize=14, fontweight='bold')
plt.ylabel('Score (%)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('metrics_chart.png', dpi=150)
plt.show()
print("   Metrics chart saved to: metrics_chart.png")

print("\nEvaluation complete!")