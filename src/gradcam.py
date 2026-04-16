import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class ImageOnlyWrapper(nn.Module):
    """Wraps the multimodal model to accept only image input for GradCAM."""

    def __init__(self, multimodal_model, fixed_tabular: torch.Tensor):
        super().__init__()
        self.model = multimodal_model
        self.fixed_tabular = fixed_tabular

    def forward(self, image):
        tabular = self.fixed_tabular.expand(image.size(0), -1)
        return self.model(image, tabular)


def find_target_layer(model):
    """Find the last convolutional layer in EfficientNet backbone."""
    # For timm's EfficientNet-B0, the last conv layer is conv_head
    if hasattr(model.backbone, "conv_head"):
        return model.backbone.conv_head
    # Fallback: find the last Conv2d layer
    last_conv = None
    for module in model.backbone.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def generate_gradcam(
    model,
    image_tensor: torch.Tensor,
    tabular_tensor: torch.Tensor,
    original_image_np: np.ndarray,
    predicted_class: int | None = None,
) -> np.ndarray:
    """
    Generate GradCAM visualization.

    Args:
        model: Multimodal model (on CPU or GPU)
        image_tensor: Preprocessed image tensor [1, 3, 224, 224]
        tabular_tensor: Tabular features [18]
        original_image_np: Original image as float32 numpy array [H, W, 3] in [0, 1]
        predicted_class: Class index to visualize. If None, uses the predicted class.

    Returns:
        Overlay image as uint8 numpy array [H, W, 3]
    """
    model.eval()
    device = next(model.parameters()).device

    # Wrap model
    wrapper = ImageOnlyWrapper(model, tabular_tensor.unsqueeze(0).to(device))
    target_layer = find_target_layer(model)

    if target_layer is None:
        # Return original image if no target layer found
        return (original_image_np * 255).astype(np.uint8)

    cam = GradCAM(model=wrapper, target_layers=[target_layer])

    # Determine target class
    if predicted_class is None:
        with torch.no_grad():
            logits = wrapper(image_tensor.unsqueeze(0).to(device))
            predicted_class = logits.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0).to(device), targets=targets)
    grayscale_cam = grayscale_cam[0]  # [H, W]

    # Resize original image to match
    from PIL import Image
    h, w = grayscale_cam.shape
    img_resized = np.array(Image.fromarray(
        (original_image_np * 255).astype(np.uint8)
    ).resize((w, h))).astype(np.float32) / 255.0

    overlay = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
    return overlay
