import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

from src.config import (
    PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    CLASS_NAMES, NUM_CLASSES,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    SCHEDULER_PATIENCE, EARLY_STOP_PATIENCE, NUM_WORKERS,
    RANDOM_SEED,
)
from src.dataset import HAM10000Dataset, get_train_transform, get_val_transform
from src.model import MultimodalSkinLesionNet, ImageOnlyModel


def compute_class_weights(train_csv: str) -> torch.Tensor:
    import pandas as pd
    df = pd.read_csv(train_csv)
    class_counts = df["dx"].value_counts()
    total = len(df)
    weights = []
    for cls in CLASS_NAMES:
        w = total / (NUM_CLASSES * class_counts.get(cls, 1))
        weights.append(w)
    # Normalize and cap
    min_w = min(weights)
    weights = [min(w / min_w, 20.0) for w in weights]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, tabular, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images, tabular)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, bal_acc


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, tabular, labels in tqdm(loader, desc="  Val", leave=False):
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images, tabular)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, bal_acc


def train(model_type: str = "multimodal"):
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    # Datasets
    train_csv = str(PROCESSED_DIR / "train.csv")
    val_csv = str(PROCESSED_DIR / "val.csv")
    train_dataset = HAM10000Dataset(train_csv, transform=get_train_transform())
    val_dataset = HAM10000Dataset(val_csv, transform=get_val_transform())

    num_workers = NUM_WORKERS if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Model
    if model_type == "multimodal":
        model = MultimodalSkinLesionNet()
        print("Training: Multimodal Fusion Model")
    else:
        model = ImageOnlyModel()
        print("Training: Image-Only Baseline Model")
    model = model.to(device)

    # Loss with class weights
    class_weights = compute_class_weights(train_csv).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: {dict(zip(CLASS_NAMES, class_weights.cpu().tolist()))}")

    # Optimizer with differential learning rates
    optimizer = torch.optim.AdamW(
        model.get_param_groups(LEARNING_RATE),
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHEDULER_PATIENCE, factor=0.5,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_bal_acc": [], "val_bal_acc": []}
    checkpoint_name = f"best_model_{model_type}.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_bal_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp,
        )
        val_loss, val_bal_acc = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[1]["lr"]
        print(
            f"  Train Loss: {train_loss:.4f} | Train Bal Acc: {train_bal_acc:.4f}\n"
            f"  Val   Loss: {val_loss:.4f} | Val   Bal Acc: {val_bal_acc:.4f}\n"
            f"  LR: {current_lr:.6f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_bal_acc"].append(train_bal_acc)
        history["val_bal_acc"].append(val_bal_acc)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / checkpoint_name)
            print(f"  -> Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Save training history
    history_name = f"training_history_{model_type}.json"
    with open(RESULTS_DIR / history_name, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {CHECKPOINT_DIR / checkpoint_name}")
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["multimodal", "image_only"], default="multimodal")
    args = parser.parse_args()
    train(model_type=args.model)
