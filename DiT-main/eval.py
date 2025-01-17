import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

# DCGAN / diffusion
# transform = transforms.Compose(
#         [
#             transforms.Resize((32, 32)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5])
#         ])

# DiT

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

dataset = datasets.ImageFolder('./data', transform=transform)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# 将数据集转换为 NumPy 数组
images = []
labels = []

for img, label in data_loader:
    # img.shape: [1, 3, 256, 256] -> 转换为 [256, 256, 3]
    img = img.squeeze(0).permute(1, 2, 0).numpy()  # 将张量转换为 NumPy 数组
    img = (img * 255).astype(np.uint8)  # 还原到 [0, 255] 范围
    images.append(img)
    labels.append(label.item())

images = np.array(images)  # 转换为 NumPy 数组 (N, H, W, C)
labels = np.array(labels)  # 转换为 NumPy 数组 (N,)

# 保存为 .npz 文件
np.savez('data_images.npz', images=images, labels=labels)

print(f"Saved {len(images)} images and labels to 'data_images.npz'")