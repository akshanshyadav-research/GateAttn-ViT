#!/usr/bin/env python3
# Original notebook: vision_pipeline/Untitled.ipynb
# Archival export. See docs/ARCHIVE.md before executing.
if __name__ != "__main__":
    raise ImportError("Archival experiment: use the gateattn package for imports.")

# %% Cell 0 (code)
import torch
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig

# Load config with attention output enabled
model_name = "google/vit-base-patch16-224-in21k"
config = ViTConfig.from_pretrained(model_name, output_attentions=True)
model = ViTModel.from_pretrained(model_name, config=config)
model.eval()

# 2. Load and preprocess image
img = Image.open("/home/viraj/my_project/val/val2/n01443537/ILSVRC2012_val_00000994.JPEG").convert("RGB")
inputs = feature_extractor(images=img, return_tensors="pt")

# 3. Forward pass (get attentions)
with torch.no_grad():
    outputs = model(**inputs)
    # outputs.attentions is a tuple: one tensor per layer, each shaped (batch, heads, tokens, tokens)
    # we’ll use the last layer
    attn = outputs.attentions[-1]  # shape [1, num_heads, num_tokens, num_tokens]

# 4. Reduce: average over heads, take CLS token’s attentions to patches
attn_avg = attn.mean(dim=1)[0]      # [num_tokens, num_tokens]
cls_attn = attn_avg[0, 1:]          # skip the CLS→CLS self-attn
cls_attn = cls_attn.reshape(14, 14) # ViT-base uses 16×16 patches on 224×224 → 14×14 grid

# 5. Normalize heatmap
heatmap = cls_attn.cpu().numpy()
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

# 6. Upsample to image size and apply colormap
img_np = np.array(img)
h, w, _ = img_np.shape
heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
heatmap_uint8 = np.uint8(255 * heatmap_resized)
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

# 7. Overlay
overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

# 8. Display
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(img_np);         plt.title("Original");   plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap_resized, cmap="jet"); plt.title("Attention Map"); plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay);        plt.title("Overlay");    plt.axis("off")

plt.tight_layout()
plt.show()


