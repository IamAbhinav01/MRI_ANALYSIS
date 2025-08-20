import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# ---------------- Streamlit Title ----------------
st.set_page_config(page_title="MRI Analysis", page_icon="🧠", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 MRI Tumor Analysis with Grad-CAM</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- File uploader ----------------
file = st.file_uploader("📂 Upload an MRI Image", type=["jpg", "jpeg", "png"])

# ---------------- Define Model ----------------
class MRINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights=None)  # custom trained model
        self.base.fc = nn.Linear(self.base.fc.in_features, 4)

    def forward(self, x):
        return self.base(x)

# Load trained model
model = MRINet()
model.load_state_dict(torch.load("mri_classifier.pth", map_location="cpu"))
model.eval()

# ---------------- Grad-CAM ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

# ---------------- Preprocessing ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ---------------- Classes ----------------
classes = ["🧩 Glioma Tumor", "🧩 Meningioma Tumor", "✅ No Tumor", "🧩 Pituitary Tumor"]

# ---------------- Prediction & Grad-CAM ----------------
if file:
    # Load and preprocess image
    img = Image.open(file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    # 🔍 Prediction Card
    st.markdown("### 🔍 Prediction Result")
    st.success(f"**Prediction:** {classes[predicted.item()]} \n\n**Confidence:** {confidence.item()*100:.2f}%")

    # 📊 Confidence as progress bars
    st.markdown("### 📊 Confidence Scores")
    for i, cls in enumerate(classes):
        st.write(f"{cls}")
        st.progress(float(probs[0][i].item()))

    # 🔥 Grad-CAM
    target_layer = model.base.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    cam, pred_class = gradcam.generate(img_tensor, class_idx=predicted.item())

    # Overlay
    img_np = np.array(img.resize((224, 224))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = cv2.addWeighted(np.float32(img_np), 0.5, heatmap, 0.5, 0)

    # Show side by side
    st.markdown("### 🎨 Grad-CAM Visualization")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_np, caption="🖼️ Original Image", use_container_width=True)
    with col2:
        st.image(cam, caption="🔥 Heatmap", use_container_width=True, clamp=True)
    with col3:
        st.image(overlay, caption=f"Overlay ({classes[pred_class]})", use_container_width=True)

    # Optional: Download overlay
    overlay_bgr = (overlay * 255).astype(np.uint8)
    is_success, im_buf_arr = cv2.imencode(".png", overlay_bgr)
    if is_success:
        st.download_button(label="⬇️ Download Overlay Image", data=im_buf_arr.tobytes(),
                           file_name="gradcam_overlay.png", mime="image/png")
