import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, cohen_kappa_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    CLASS_NAMES, CLASS_LABELS, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS,
)
from src.dataset import HAM10000Dataset, get_val_transform
from src.model import MultimodalSkinLesionNet, ImageOnlyModel


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, tabular, labels in loader:
        images = images.to(device)
        tabular = tabular.to(device)
        logits = model(images, tabular)
        probs = torch.softmax(logits, dim=1)

        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds, save_path, model_type="multimodal"):
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix ({model_type})")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")

    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title(f"Normalized Confusion Matrix ({model_type})")
    axes[1].set_ylabel("True")
    axes[1].set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_history(history_path, save_path):
    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_bal_acc"], "b-", label="Train")
    axes[1].plot(epochs, history["val_bal_acc"], "r-", label="Validation")
    axes[1].set_title("Balanced Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate(model_type: str = "multimodal"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if model_type == "multimodal":
        model = MultimodalSkinLesionNet()
    else:
        model = ImageOnlyModel()

    checkpoint_path = CHECKPOINT_DIR / f"best_model_{model_type}.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)

    # Load test data
    test_dataset = HAM10000Dataset(
        str(PROCESSED_DIR / "test.csv"), transform=get_val_transform(),
    )
    num_workers = NUM_WORKERS if device.type == "cuda" else 0
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Get predictions
    labels, preds, probs = get_predictions(model, test_loader, device)

    # Metrics
    bal_acc = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    kappa = cohen_kappa_score(labels, preds)

    try:
        roc_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    # Melanoma recall (clinically most important)
    mel_idx = CLASS_NAMES.index("mel")
    mel_mask = labels == mel_idx
    mel_recall = (preds[mel_mask] == mel_idx).mean() if mel_mask.sum() > 0 else 0.0

    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=4,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({model_type})")
    print(f"{'='*60}")
    print(f"Balanced Accuracy:  {bal_acc:.4f}")
    print(f"Macro F1-Score:     {macro_f1:.4f}")
    print(f"Weighted F1-Score:  {weighted_f1:.4f}")
    print(f"Cohen's Kappa:      {kappa:.4f}")
    print(f"ROC-AUC (macro):    {roc_auc:.4f}")
    print(f"Melanoma Recall:    {mel_recall:.4f}")
    print(f"\n{report}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "model_type": model_type,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
        "roc_auc_macro": roc_auc,
        "melanoma_recall": mel_recall,
    }
    with open(RESULTS_DIR / f"metrics_{model_type}.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(RESULTS_DIR / f"classification_report_{model_type}.txt", "w") as f:
        f.write(f"Evaluation Results ({model_type})\n")
        f.write(f"{'='*60}\n")
        f.write(f"Balanced Accuracy:  {bal_acc:.4f}\n")
        f.write(f"Macro F1-Score:     {macro_f1:.4f}\n")
        f.write(f"Weighted F1-Score:  {weighted_f1:.4f}\n")
        f.write(f"Cohen's Kappa:      {kappa:.4f}\n")
        f.write(f"ROC-AUC (macro):    {roc_auc:.4f}\n")
        f.write(f"Melanoma Recall:    {mel_recall:.4f}\n\n")
        f.write(report)

    # Plots
    plot_confusion_matrix(
        labels, preds,
        RESULTS_DIR / f"confusion_matrix_{model_type}.png",
        model_type,
    )

    history_path = RESULTS_DIR / f"training_history_{model_type}.json"
    if history_path.exists():
        plot_training_history(
            history_path,
            RESULTS_DIR / f"training_curves_{model_type}.png",
        )

    print(f"\nResults saved to {RESULTS_DIR}/")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["multimodal", "image_only"], default="multimodal")
    args = parser.parse_args()
    evaluate(model_type=args.model)
