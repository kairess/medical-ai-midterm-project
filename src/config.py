from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# === Dataset ===
IMAGE_SIZE = 224
NUM_CLASSES = 7
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_LABELS = {
    "akiec": "광선각화증 (Actinic Keratoses)",
    "bcc": "기저세포암 (Basal Cell Carcinoma)",
    "bkl": "양성 각화증 (Benign Keratosis)",
    "df": "피부섬유종 (Dermatofibroma)",
    "mel": "멜라노마 (Melanoma)",
    "nv": "멜라닌세포 모반 (Melanocytic Nevi)",
    "vasc": "혈관 병변 (Vascular Lesions)",
}
CLASS_MALIGNANCY = {
    "akiec": True,
    "bcc": True,
    "bkl": False,
    "df": False,
    "mel": True,
    "nv": False,
    "vasc": False,
}

# === Tabular Feature Encoding ===
LOCALIZATION_CATEGORIES = [
    "abdomen", "back", "chest", "ear", "face", "foot",
    "genital", "hand", "lower extremity", "neck",
    "scalp", "trunk", "upper extremity", "unknown",
]
SEX_CATEGORIES = ["female", "male", "unknown"]
NUM_TABULAR_FEATURES = 1 + len(SEX_CATEGORIES) + len(LOCALIZATION_CATEGORIES)  # 18

# === Training ===
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 5
EARLY_STOP_PATIENCE = 7
NUM_WORKERS = 4

# === Model Architecture ===
BACKBONE_NAME = "efficientnet_b0"
BACKBONE_FEATURE_DIM = 1280
IMAGE_FEATURE_DIM = 256
TABULAR_HIDDEN_DIM = 64
FUSION_HIDDEN_DIM = 128
DROPOUT_RATE = 0.3
FREEZE_BACKBONE_RATIO = 0.7

# === Data Splits ===
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
