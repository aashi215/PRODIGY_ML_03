"""
Task-03: SVM Image Classifier — Cats vs Dogs
Dataset: https://www.kaggle.com/c/dogs-vs-cats/data

How to run:
1. Download the Kaggle dataset and unzip it so you have a folder like:
       dogs-vs-cats/train/   (contains cat.0.jpg, dog.0.jpg, ...)
2. Set DATASET_PATH below to that folder path.
3. pip install numpy scikit-learn pillow matplotlib seaborn tqdm
4. python svm_cats_dogs.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURATION  (edit these as needed)
# ─────────────────────────────────────────────
DATASET_PATH = r"C:\Users\kusha\OneDrive\Desktop\task 3\dogs-vs-cats-redux-kernels-edition\train\train"
IMG_SIZE     = (64, 64)                # resize target; bigger = better but slower
MAX_IMAGES   = 2000                    # total images to load (1000 cats + 1000 dogs)
TEST_SIZE    = 0.20                    # 80/20 train-test split
PCA_COMPONENTS = 150                   # dimensionality reduction before SVM
RANDOM_STATE = 42
# ─────────────────────────────────────────────


# ══════════════════════════════════════════════
#  1. DATA LOADING
# ══════════════════════════════════════════════
def load_images(folder: str, max_per_class: int = 1000):
    """Load and resize images; return flat numpy arrays + labels."""
    X, y = [], []
    class_counts = {"cat": 0, "dog": 0}

    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"\n[INFO] Found {len(files)} total files in '{folder}'")

    for fname in tqdm(files, desc="Loading images"):
        label_str = fname.split(".")[0].lower()          # "cat" or "dog"
        if label_str not in class_counts:
            continue
        if class_counts[label_str] >= max_per_class:
            continue

        try:
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
            X.append(np.array(img, dtype=np.float32).flatten() / 255.0)
            y.append(0 if label_str == "cat" else 1)
            class_counts[label_str] += 1
        except Exception as e:
            print(f"  [WARN] Skipping {fname}: {e}")

        if sum(class_counts.values()) >= max_per_class * 2:
            break

    print(f"[INFO] Loaded — cats: {class_counts['cat']}, dogs: {class_counts['dog']}")
    return np.array(X), np.array(y)


# ══════════════════════════════════════════════
#  2. PREPROCESSING  (PCA + Scaling)
# ══════════════════════════════════════════════
def preprocess(X_train, X_test):
    print(f"\n[INFO] Raw feature size: {X_train.shape[1]}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    n_components = min(PCA_COMPONENTS, X_train_s.shape[0], X_train_s.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p  = pca.transform(X_test_s)

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"[INFO] PCA reduced to {n_components} components "
          f"({explained:.1f}% variance explained)")

    return X_train_p, X_test_p, scaler, pca


# ══════════════════════════════════════════════
#  3. MODEL TRAINING
# ══════════════════════════════════════════════
def train_svm(X_train, y_train, tune_hyperparams: bool = False):
    """Train SVM with optional GridSearchCV hyperparameter tuning."""

    if tune_hyperparams:
        print("\n[INFO] Running GridSearchCV (this may take a few minutes)…")
        param_grid = {
            "C":      [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma":  ["scale", "auto"],
        }
        grid = GridSearchCV(
            SVC(probability=True, random_state=RANDOM_STATE),
            param_grid,
            cv=3, scoring="accuracy", n_jobs=-1, verbose=1
        )
        grid.fit(X_train, y_train)
        print(f"[INFO] Best params: {grid.best_params_}")
        return grid.best_estimator_

    else:
        print("\n[INFO] Training SVM (RBF kernel, C=10)…")
        model = SVC(C=10, kernel="rbf", gamma="scale",
                    probability=True, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        return model


# ══════════════════════════════════════════════
#  4. EVALUATION & PLOTS
# ══════════════════════════════════════════════
def evaluate(model, X_test, y_test, X_train_raw, y_train):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print("\n" + "═"*50)
    print(f"  TEST ACCURACY : {acc*100:.2f}%")
    print("═"*50)
    print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

    # ── Confusion Matrix ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SVM — Cats vs Dogs Classification", fontsize=15, fontweight="bold")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix")

    # ── Per-class Precision/Recall/F1 bar chart ──
    report = classification_report(y_test, y_pred,
                                   target_names=["Cat", "Dog"],
                                   output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(metrics))
    width = 0.35
    axes[1].bar(x - width/2, [report["Cat"][m]  for m in metrics], width, label="Cat",  color="#4C72B0")
    axes[1].bar(x + width/2, [report["Dog"][m]  for m in metrics], width, label="Dog",  color="#DD8452")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Precision", "Recall", "F1-Score"])
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Per-Class Metrics")
    axes[1].legend()
    axes[1].axhline(acc, color="gray", linestyle="--", linewidth=1,
                    label=f"Overall Acc: {acc*100:.1f}%")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("svm_evaluation.png", dpi=150)
    print("[INFO] Saved evaluation plot → svm_evaluation.png")
    plt.show()


# ══════════════════════════════════════════════
#  5. SAMPLE PREDICTIONS GRID
# ══════════════════════════════════════════════
def show_sample_predictions(model, pca, scaler, folder, n=12):
    """Display a grid of predictions on random images."""
    files  = [f for f in os.listdir(folder)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    sample = np.random.choice(files, size=n, replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle("Sample Predictions", fontsize=14, fontweight="bold")

    for ax, fname in zip(axes.flat, sample):
        true_label = fname.split(".")[0].lower()
        img_path   = os.path.join(folder, fname)

        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        feat = np.array(img, dtype=np.float32).flatten() / 255.0
        feat_s = scaler.transform([feat])
        feat_p = pca.transform(feat_s)
        pred   = model.predict(feat_p)[0]
        prob   = model.predict_proba(feat_p)[0].max()

        pred_label = "Dog" if pred == 1 else "Cat"
        color      = "green" if pred_label.lower() == true_label else "red"

        ax.imshow(img)
        ax.set_title(f"Pred: {pred_label} ({prob*100:.0f}%)\nTrue: {true_label.capitalize()}",
                     color=color, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150)
    print("[INFO] Saved sample grid  → sample_predictions.png")
    plt.show()


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":

    # ── 1. Load ────────────────────────────────
    per_class = MAX_IMAGES // 2
    X, y = load_images(DATASET_PATH, max_per_class=per_class)

    if len(X) == 0:
        raise FileNotFoundError(
            f"No images found in '{DATASET_PATH}'.\n"
            "Please set DATASET_PATH to the folder containing cat.*.jpg / dog.*.jpg files."
        )

    # ── 2. Split ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 3. Preprocess ──────────────────────────
    X_train_p, X_test_p, scaler, pca = preprocess(X_train, X_test)

    # ── 4. Train ───────────────────────────────
    # Set tune_hyperparams=True to run GridSearchCV (slower but potentially better)
    model = train_svm(X_train_p, y_train, tune_hyperparams=False)

    # ── 5. Evaluate ────────────────────────────
    evaluate(model, X_test_p, y_test, X_train, y_train)

    # ── 6. Sample predictions ──────────────────
    show_sample_predictions(model, pca, scaler, DATASET_PATH, n=12)

    print("\n[DONE] All outputs saved.")