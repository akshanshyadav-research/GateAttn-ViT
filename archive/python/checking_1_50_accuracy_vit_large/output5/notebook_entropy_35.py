#!/usr/bin/env python3
# Original notebook: checking_1_50_accuracy_vit_large/output5/notebook_entropy_35.ipynb
# Archival export. See docs/ARCHIVE.md before executing.
if __name__ != "__main__":
    raise ImportError("Archival experiment: use the gateattn package for imports.")

# %% Cell 0 (code)
patch_drop_blocks=[]
drop_rate=0.00
attention_pruning= 0.00
drop_patch_percentage_entropy=35

# %% Cell 1 (code)
import matplotlib.pyplot as plt
from time import time
import numpy as np
import os
import cv2
import json
import torch
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
import torch
import torch.nn as nn
import numpy as np
import os
import re
import timm

# %% Cell 2 (code)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
img_folder = "/home/viraj/my_project/val/val2/"
txt_path = f"vit_base_entropy_grey_dropblock{patch_drop_blocks}_droprate{drop_rate:.2f}_attnprune{attention_pruning:.2f}.txt"
json_file = "/home/viraj/my_project/imagenet_class_index.json"
# line_r=0
 

# %% Cell 3 (code)
# pretrained_vit = vit_large_patch16_224(pretrained=True).to(device)

pretrained_vit = timm.create_model('vit_large_patch16_224', pretrained=True).to(device)

# %% Cell 4 (code)
def plot(image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.show()

# %% Cell 5 (code)
def copy_image(img):
    """
    Create a copy of the input image without using any libraries.

    Args:
    - img: Input image (list of lists representing pixel values)

    Returns:
    - img_copy: A copy of the input image
    """
    # Determine the dimensions of the image
    height = len(img)
    width = len(img[0])

    # Create an empty list to store the copied image
    img_copy = []

    # Iterate over each row of the image
    for i in range(height):
        # Create a copy of the current row
        row_copy = img[i][:]  # or row_copy = list(img[i]) for deep copy
        # Append the copied row to the copied image
        img_copy.append(row_copy)

    return img_copy

# %% Cell 6 (code)
def bgr_to_rgb(img):
    """
    Convert an image from BGR to RGB color space.

    Args:
    - img: Input image in BGR format (numpy array)

    Returns:
    - img_rgb: Image converted to RGB color space
    """
    # Reverse the order of color channels
    img_rgb = img[:, :, ::-1]

    return img_rgb

# %% Cell 7 (code)
def resize_nearest_neighbor(image, size):
    original_height = len(image)
    original_width = len(image[0])
    new_width, new_height = size

    if isinstance(image[0][0], list) or isinstance(image[0][0], np.ndarray):
        # For color images
        resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                # Calculate the nearest pixel in the original image
                x = int(j * original_width / new_width)
                y = int(i * original_height / new_height)
                resized_image[i, j] = image[y][x]
    else:
        # For grayscale images
        resized_image = np.zeros((new_height, new_width), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                # Calculate the nearest pixel in the original image
                x = int(j * original_width / new_width)
                y = int(i * original_height / new_height)
                resized_image[i, j] = image[y][x]
    
    return np.array(resized_image)

# %% Cell 8 (code)
def save_average_percentage(file_path, average_percent, accuracy, processed_image, w):
    with open(file_path, 'a') as file:
        file.write(f"Epoch: Average percentage_patch_drop: {average_percent}, Accuracy: {accuracy}, processed_image: {processed_image}, scan: {w}\n")

# %% Cell 9 (code)
def load_mapping_file(json_file):
    with open(json_file, 'r') as f:
        mapping_data = json.load(f)
    return mapping_data

# %% Cell 10 (code)
import torchvision.transforms as transforms
def transform_image(image, device):

    image=transforms.ToTensor()(image)
    image= image.unsqueeze(0)
    image=image.to(device)    
    

    
    return image

# %% Cell 11 (code)
mapping_data = load_mapping_file(json_file)

def get_key_by_value(predicted_index, class_name):
    
    for key, values in mapping_data.items():
        key = int(key)
        
        if class_name in values:
            # print(key, predicted_index, "key and predicted_index")
            # print(f"Found class_name '{class_name}' in values: {values}")
            if int(key) == predicted_index:
                return True
    return False

# %% Cell 12 (code)
patch_drop_percentage = []
def average_patch_drop(mask):
    count_ones = mask.count(1)
    count_ones=count_ones-1
    y = (((196 - count_ones ) / 196) * 100)
    patch_drop_percentage.append(y)
    

# %% Cell 13 (code)
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer, Block
import numpy as np
import torch.nn.functional as F
class CustomAttention(nn.Module):
    def __init__(self, attn, num_heads):
        super(CustomAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = attn.scale
        self.qkv = attn.qkv
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        mask_attention = None
        avg_attn = attn.mean(dim=1)
        cls_token_attn = avg_attn[:, 0, :] 
        threshold = torch.quantile(cls_token_attn, attention_pruning , dim=-1, keepdim=True)
        mask_attention = cls_token_attn >= threshold
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, mask_attention


class PatchDropout(nn.Module):
    def __init__(self):
        super().__init__()
        # self.drop_rate = drop_rate  # Drop 20% of the tokens

    def forward(self, x, mask):
        # batch_size, seq_len, embedding_dim = x.shape
        # assert seq_len == 197, "Expected sequence length of 197, but got {}".format(seq_len)
        # # Extract CLS token (first token)
        # cls_token = x[:, 0, :]  # Shape: (batch_size, embedding_dim)

        # # Compute cosine similarity with all other tokens
        # similarity_scores = F.cosine_similarity(x[:, 1:, :], cls_token.unsqueeze(1), dim=-1)  # (batch_size, 196)

        # # Determine threshold to drop top 20% most similar tokens
        # k = int(seq_len * drop_rate)  # 20% of 196 tokens (~39 tokens)
        # threshold_values, _ = torch.kthvalue(similarity_scores, seq_len - k - 1, dim=1, keepdim=True)  # (batch_size, 1)

        # # Create a mask to **remove top 20% (high similarity scores)**
        # keep_mask = similarity_scores < threshold_values  # (batch_size, 196)
        # # Append True for CLS token (always keep it)
        # keep_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=x.device), keep_mask], dim=1)  # (batch_size, 197)
        # # Apply mask and return filtered tensor
        # x_filtered = x[keep_mask].view(batch_size, -1, embedding_dim)  # Adjust shape after filtering
        # # print(x_filtered.shape)
        # return x_filtered
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        indices = torch.nonzero(mask_tensor).squeeze() 
        tensor2 = x[:, indices, :]
        # print(tensor2.shape, "tensor2")
        return tensor2


        # return x

class CustomBlock(nn.Module):
    def __init__(self, block):
        super(CustomBlock, self).__init__()
        self.norm1 = block.norm1
        self.attn = CustomAttention(block.attn, block.attn.num_heads)
        self.ls1 = block.ls1
        self.drop_path1 = block.drop_path1
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls2 = block.ls2
        self.drop_path2 = block.drop_path2

    def forward(self, x, apply_patch_drop=False):

        
        x_residual = self.norm1(x)
        x_residual, mask_attention = self.attn(x_residual)
        x = x + self.drop_path1(self.ls1(x_residual))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        if apply_patch_drop:
            
            cls_token = x[:, 0].unsqueeze(1)  # Keep the class token
            other_tokens = x[:, 1:]
            selected_tokens = other_tokens[mask_attention[:, 1:]] 
            selected_tokens = selected_tokens.unsqueeze(0)
            x = torch.cat((cls_token, selected_tokens), dim=1)
            
        return x

class CustomVisionTransformer(VisionTransformer):
    def __init__(self, patch_drop_blocks=None, **kwargs):
        super(CustomVisionTransformer, self).__init__(**kwargs)
        self.patch_drop_blocks = patch_drop_blocks if patch_drop_blocks is not None else []
        self.patch_drop = PatchDropout()
        self.blocks = nn.Sequential(*[
            CustomBlock(block) if i in self.patch_drop_blocks else block
            for i, block in enumerate(self.blocks)
        ])

    def forward(self, x, mask=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # class token
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if mask is not None:
            x = self.patch_drop(x, mask)

        for i, block in enumerate(self.blocks):
            if isinstance(block, CustomBlock):
                x = block(x, apply_patch_drop=True)
                # print(i, x.shape)
            else:
                x = block(x)
                # print(i)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

# Example usage
my_array = np.random.randint(1, 2, size=197)  # Updated to have values 0 or 1
custom_vit = CustomVisionTransformer(
    img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
    norm_layer=nn.LayerNorm, num_classes=1000, patch_drop_blocks= patch_drop_blocks
).to(device)


# Load pretrained weights
# pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
custom_vit.load_state_dict(pretrained_vit.state_dict(), strict=True)

# Forward pass with input tensor and masks
input_tensor = torch.randn(1, 3, 224, 224).to(device)
output = custom_vit(input_tensor, mask=my_array)
print(output.shape)  # Should be [1, 1000]


# %% Cell 14 (code)
from qtorch.quant import Quantizer, quantizer
from qtorch import FixedPoint
# Define the fixed-point quantizer
weight_quantizer = quantizer(forward_number=FixedPoint(16, 8), forward_rounding="nearest")
act_quantizer = quantizer(forward_number=FixedPoint(16, 8), forward_rounding="nearest")

# %% Cell 15 (code)
import torch
import torch.nn as nn

# Define the QuantizedVisionTransformer class
class QuantizedVisionTransformer(nn.Module):
    def __init__(self, model, weight_quantizer, act_quantizer):
        super(QuantizedVisionTransformer, self).__init__()
        self.model = model
        self.weight_quantizer = weight_quantizer
        self.act_quantizer = act_quantizer
        self.apply(self._quantize_weights)

    def _quantize_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data = self.weight_quantizer(module.weight.data)
            if module.bias is not None:
                module.bias.data = self.weight_quantizer(module.bias.data)
        elif isinstance(module, nn.Conv2d):
            module.weight.data = self.weight_quantizer(module.weight.data)
            if module.bias is not None:
                module.bias.data = self.weight_quantizer(module.bias.data)

    def forward(self, x, mask):
        x = self.act_quantizer(x)
        # for name, module in self.model.named_children():
        #     x = module(x, mask) if 'block' in name else module(x)
        #     if isinstance(module, (nn.Linear, nn.Conv2d)):
        #         x = self.act_quantizer(x)
        return self.model(x, mask)

# Example usage:
# Assume `vision_transformer_model` is your pretrained model
# and `weight_quantizer` and `act_quantizer` are your quantization functions
quantized_model = QuantizedVisionTransformer(custom_vit, weight_quantizer, act_quantizer)



# %% Cell 16 (code)
def check_quantization(model):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Data type: {param.dtype}")
        print(f"Values: {param.view(-1)[:10]}")  # Print the first 10 values
        print(f"Storage size (bytes): {param.numel() * param.element_size()}")
        print()



# Check quantized weights and biases
# check_quantization(quantized_model)

# %% Cell 17 (code)
output = quantized_model(input_tensor, mask=my_array)

# %% Cell 18 (code)
import cv2
import numpy as np
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
keep_percentage= 100-drop_patch_percentage_entropy
percentage_entropy = []

def process_image_and_create_entropy_binary_array(image, keep_percentage):
    """
    Processes an image by dividing it into 16x16 patches, calculating entropy for each patch,
    and selecting only the top percentage of patches with the highest entropy.

    Args:
    - image (ndarray): The input image as a NumPy array.
    - fixed_percentage (float): Percentage of top entropy patches to keep (set to 1).

    Returns:
    - binary_array (list): Binary array with 1s for top entropy patches, 0s elsewhere.
    """
    # Step 1: Resize image to 224x224
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image, (224, 224))

    # Step 2: Divide the image into 16x16 patches
    patch_size = 16
    patches = []
    for i in range(0, 224, patch_size):
        for j in range(0, 224, patch_size):
            patch = image_resized[i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    # Step 3: Compute entropy for each patch
    entropy_values = [shannon_entropy(patch) for patch in patches]

    # Step 4: Determine how many patches to keep
    num_patches = len(entropy_values)
    num_fixed = max(1, int((keep_percentage / 100) * num_patches))

    # Step 5: Get indices of top entropy patches
    top_indices = np.argsort(entropy_values)[-num_fixed:]

    # Step 6: Create binary array with 1 for selected patches
    binary_array = [0] * num_patches
    for idx in top_indices:
        binary_array[idx] = 1

    # Step 7: Track percentage of retained patches
    zero_count = binary_array.count(0)
    zero_percentage = (zero_count / num_patches) * 100
    percentage_entropy.append(zero_percentage)
    # print(zero_percentage, "zero_percentage")

    # Step 8: Insert 1 at the beginning (e.g., for CLS token)
    binary_array.insert(0, 1)

    return binary_array


# %% Cell 19 (code)
def main_processing(image):
    resized_image =  resize_nearest_neighbor(image, (224, 224))
    resized_image = bgr_to_rgb(resized_image)
    # gray_image = rgb_to_gray(resized_image)
    resized_image = resized_image.copy()                                        
    image_tensor= transform_image(resized_image, device)
    mask= process_image_and_create_entropy_binary_array(image, keep_percentage)
    
    # mask = np.random.randint(1, 2, size=197) 
    # print(mask)
    
    return image_tensor, mask


# %% Cell 20 (code)


# %% Cell 21 (code)
def save_percentage(file_path, average_percent1, average_percent2, average_percent3 ,average_percent4, average, count):
    with open(file_path, 'a') as file:
        file.write(f"Epoch: Average percentage_patch_drop: { average_percent1}, { average_percent2}, { average_percent3}, { average_percent4}, average: {average}, count{count}\n")

# %% Cell 22 (code)
# Image processing function
from PIL import Image
from PIL import ImageOps
def apply_mask_with_white_shade(image_path, mask_tensor, opacity=0.4, border_size=1):
    # mask = mask_tensor.to('cpu').numpy().flatten()

    # Load and resize image
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_np = np.array(image).astype(np.float32)

    patch_size = 16
    output_image = image_np.copy()
    mask=mask_tensor
    # assert mask.shape[0] == 196, "Mask must have 196 elements (14x14 patches)"

    idx = 0
    for i in range(0, 224, patch_size):
        for j in range(0, 224, patch_size):
            if not mask[idx]:
                # Blend the patch with white using the specified opacity
                original_patch = output_image[i:i+patch_size, j:j+patch_size, :]
                white_patch = np.ones_like(original_patch) * 255.0
                blended_patch = (1 - opacity) * original_patch + opacity * white_patch
                output_image[i:i+patch_size, j:j+patch_size, :] = blended_patch
            idx += 1

    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    result_image = Image.fromarray(output_image)

    # Add black border
    result_image_with_border = ImageOps.expand(result_image, border=border_size, fill='black')

    
    return result_image_with_border

# # ---- Example usage ----
# # image_path = mask  # Update with your actual image path
# result_image = apply_mask_with_white_shade(image_path, mask_tensor, opacity=0.70, border_size=1)

# # Display the result
# plt.figure(figsize=(6, 6))
# plt.imshow(result_image)
# plt.axis('off')
# # plt.title("Masked Image with White Overlay and Black Border")
# plt.show()

# %% Cell 23 (code)
def convert_to_canvas(img_folder, model, device):
    correct_predictions = 0
    total_images = 0
    processed_images = 0
    
    for subfolder_name in os.listdir(img_folder):
        subfolder_path = os.path.join(img_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            # Process each image in the subfolder
            for img_name in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Load the image
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Failed to load image: {img_path}")
                        continue

                    image_tensor, fast_mask = main_processing(image)
                    
                    # outputs = custom_vit( image_tensor, mask= fast_mask)
                    outputs= quantized_model(image_tensor, mask= fast_mask)
                    predicted_idx = torch.argmax(outputs, dim=1).item()
                    match = re.search(r'val2/([^/]+)/', img_path)
                    if match:
                        class_name = match.group(1)
                    else:
                        class_name = None
                    
                    prediction_result = get_key_by_value(predicted_idx, class_name)
                    
                    if prediction_result is True:
                        correct_predictions += 1    
                    processed_images += 1

                    

                    # mask_tensor = fast_mask[1:]
                    # print(mask_tensor)
                    # result_image = apply_mask_with_white_shade(img_path, mask_tensor, opacity=0.7, border_size=1)
                    # # Display the result
                    # plt.figure(figsize=(6, 6))
                    # plt.imshow(result_image)
                    # plt.axis('off')
                    # # plt.title("Masked Image with White Overlay and Black Border")
                    # plt.show()

                    
                    # Print accuracy after processing every 50 images
                    if processed_images % 10000 == 0:
                        # w= ( sum(scan)/ len(scan))
                        accuracy_so_far = (correct_predictions / processed_images) * 100
                        # average_percent= 15_15_15_15
                        w=0
                        # average_percent = sum(patch_drop_percentage) / len(patch_drop_percentage)
                        # # print("average percentage embedding_drop " , average_percent , len(patch_drop_percentage))
                        print(f"Accuracy after processing {processed_images} images: {accuracy_so_far:.2f}%")
                        save_average_percentage(txt_path,drop_patch_percentage_entropy, accuracy_so_far, processed_images, w )
                        # print(sum(percentage_entropy) / len(percentage_entropy))
                        # print(len(percentage_entropy))
                        # print(percentage_entropy)
                    
                    
                    


# %% Cell 24 (code)
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# Example usage
print(device)

# %% Cell 25 (code)
convert_to_canvas(img_folder, pretrained_vit, device)

# %% Cell 26 (code)


