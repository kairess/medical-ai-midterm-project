import numpy as np
import torch
from PIL import Image

from src.config import CLASS_NAMES, CLASS_LABELS, CLASS_MALIGNANCY, CHECKPOINT_DIR
from src.dataset import encode_tabular, get_val_transform
from src.model import MultimodalSkinLesionNet
from src.gradcam import generate_gradcam


def load_model(device: torch.device | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalSkinLesionNet()
    checkpoint_path = CHECKPOINT_DIR / "best_model_multimodal.pth"
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    return model, device


def predict_single(
    model,
    device: torch.device,
    pil_image: Image.Image,
    age: float,
    sex: str,
    localization: str,
    with_gradcam: bool = True,
) -> dict:
    """
    Run inference on a single image with metadata.

    Returns:
        dict with keys: predicted_class, predicted_label, confidence,
        is_malignant, probabilities, gradcam_overlay (optional)
    """
    # Prepare image
    transform = get_val_transform()
    image_np = np.array(pil_image.convert("RGB"))
    original_image_float = image_np.astype(np.float32) / 255.0
    transformed = transform(image=image_np)
    image_tensor = transformed["image"]  # [3, 224, 224]

    # Prepare tabular
    tabular_tensor = encode_tabular(age, sex, localization)

    # Inference
    with torch.no_grad():
        img_batch = image_tensor.unsqueeze(0).to(device)
        tab_batch = tabular_tensor.unsqueeze(0).to(device)
        logits = model(img_batch, tab_batch)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    pred_class = CLASS_NAMES[pred_idx]

    result = {
        "predicted_class": pred_class,
        "predicted_label": CLASS_LABELS[pred_class],
        "confidence": probs[pred_idx].item(),
        "is_malignant": CLASS_MALIGNANCY[pred_class],
        "probabilities": {
            CLASS_NAMES[i]: probs[i].item() for i in range(len(CLASS_NAMES))
        },
    }

    # GradCAM
    if with_gradcam:
        try:
            overlay = generate_gradcam(
                model, image_tensor, tabular_tensor,
                original_image_float, pred_idx,
            )
            result["gradcam_overlay"] = overlay
        except Exception as e:
            print(f"GradCAM failed: {e}")
            result["gradcam_overlay"] = None

    return result
