import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DIR, PROCESSED_DIR, CLASS_NAMES,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
)


def find_image_path(image_id: str, image_dirs: list[Path]) -> str | None:
    for d in image_dirs:
        path = d / f"{image_id}.jpg"
        if path.exists():
            return str(path)
    return None


def preprocess_and_split():
    metadata_path = RAW_DIR / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Please download HAM10000 from Kaggle and place files in data/raw/"
        )

    df = pd.read_csv(metadata_path)

    # Find image directories
    image_dirs = [
        RAW_DIR / "HAM10000_images_part_1",
        RAW_DIR / "HAM10000_images_part_2",
    ]
    existing_dirs = [d for d in image_dirs if d.exists()]
    if not existing_dirs:
        # Try flat structure (all images in raw/)
        if any(RAW_DIR.glob("ISIC_*.jpg")):
            existing_dirs = [RAW_DIR]
        else:
            raise FileNotFoundError(
                "Image directories not found. Expected:\n"
                "  data/raw/HAM10000_images_part_1/\n"
                "  data/raw/HAM10000_images_part_2/"
            )

    # Map image_id to file path
    df["image_path"] = df["image_id"].apply(lambda x: find_image_path(x, existing_dirs))
    missing = df["image_path"].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} images not found, dropping these rows.")
        df = df.dropna(subset=["image_path"])

    # Clean missing values
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age)
    df["sex"] = df["sex"].fillna("unknown")
    df["localization"] = df["localization"].fillna("unknown")

    # Encode label
    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    df["label"] = df["dx"].map(label_map)

    # Split by lesion_id to prevent data leakage
    lesion_df = df.groupby("lesion_id").agg({"dx": "first"}).reset_index()

    # First split: train vs temp (val+test)
    val_test_ratio = 1.0 - TRAIN_RATIO
    train_lesions, temp_lesions = train_test_split(
        lesion_df["lesion_id"],
        test_size=val_test_ratio,
        stratify=lesion_df["dx"],
        random_state=RANDOM_SEED,
    )

    # Second split: val vs test
    temp_df = lesion_df[lesion_df["lesion_id"].isin(temp_lesions)]
    val_ratio_in_temp = VAL_RATIO / val_test_ratio
    val_lesions, test_lesions = train_test_split(
        temp_df["lesion_id"],
        test_size=1.0 - val_ratio_in_temp,
        stratify=temp_df["dx"],
        random_state=RANDOM_SEED,
    )

    train_lesions_set = set(train_lesions)
    val_lesions_set = set(val_lesions)
    test_lesions_set = set(test_lesions)

    # Verify no leakage
    assert len(train_lesions_set & val_lesions_set) == 0, "Leakage: train/val overlap"
    assert len(train_lesions_set & test_lesions_set) == 0, "Leakage: train/test overlap"
    assert len(val_lesions_set & test_lesions_set) == 0, "Leakage: val/test overlap"

    # Assign splits
    columns = ["image_path", "dx", "label", "age", "sex", "localization"]
    train_df = df[df["lesion_id"].isin(train_lesions_set)][columns]
    val_df = df[df["lesion_id"].isin(val_lesions_set)][columns]
    test_df = df[df["lesion_id"].isin(test_lesions_set)][columns]

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"Data split complete:")
    print(f"  Train: {len(train_df)} images ({len(train_lesions_set)} lesions)")
    print(f"  Val:   {len(val_df)} images ({len(val_lesions_set)} lesions)")
    print(f"  Test:  {len(test_df)} images ({len(test_lesions_set)} lesions)")
    print(f"\nClass distribution (train):")
    for cls in CLASS_NAMES:
        count = (train_df["dx"] == cls).sum()
        print(f"  {cls}: {count} ({count / len(train_df) * 100:.1f}%)")


if __name__ == "__main__":
    preprocess_and_split()
