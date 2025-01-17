import torch
import torchvision
import os
import imageio
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
def show_images(x, nrow=5):
    x = x * 0.5 + 0.5  # 还原到 [0, 1] 范围
    grid = torchvision.utils.make_grid(x, nrow=nrow)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im
def train(model, train_loader, noise_scheduler, optimizer, scaler, device, epochs, save_path):
    writer = SummaryWriter(log_dir="./logs0")
    best_loss = float('inf')
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, (images, _) in enumerate(train_loader):
            clean_images = images.to(device)
            noise = torch.randn_like(clean_images).to(device)
            batch_size = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (batch_size,), device=device
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # AMP支持
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
    writer.close()
def generate_images(model, noise_scheduler, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    sample = torch.randn(25, 3, 32, 32).to(device)
    output_image = []
    with torch.no_grad():
        for i, t in enumerate(noise_scheduler.timesteps):
            residual = model(sample, t).sample
            sample = noise_scheduler.step(residual, t, sample).prev_sample
            process_image = show_images(sample, nrow=5).resize((5 * 64, 5 * 64), resample=Image.NEAREST)
            output_image.append(process_image)
    final_image = output_image[-1]
    final_image.save(os.path.join(output_dir, 'final_image.png'))
    plt.figure(dpi=300)
    plt.imshow(final_image)
    plt.axis('off')
    plt.show()
    gif_path = os.path.join(output_dir, 'output.gif')
    frames = [np.array(img) for img in output_image]
    imageio.mimsave(gif_path, frames, fps=10)
    print(f"Generated images saved in {output_dir}")


if __name__ == '__main__':
    # 参数配置
    image_size = 32
    BATCH_SIZE = 128
    EPOCHS = 500
    LEARNING_RATE = 4e-4
    SAVE_PATH = './checkpoints'
    OUTPUT_DIR = './outputs'

    os.makedirs(SAVE_PATH, exist_ok=True)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 保证图像尺寸一致
        transforms.RandomHorizontalFlip(),  # 数据增强：随机水平翻转
        transforms.RandomRotation(10),  # 数据增强：随机旋转
        transforms.ColorJitter(brightness=0.2,  # 数据增强：颜色抖动
                               contrast=0.2,
                               saturation=0.2,
                               hue=0.1),
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.5], [0.5]),  # 归一化到 [-1, 1]
    ])

    data = datasets.ImageFolder('../DiT-main/data', transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=3,  # 每个块使用更多层
        block_out_channels=(128, 256, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=200, beta_schedule='squaredcos_cap_v2')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()  # AMP支持

    # 开始训练
    train(model, train_loader, noise_scheduler, optimizer, scaler, device, EPOCHS, SAVE_PATH)

    # 生成图像
    generate_images(model, noise_scheduler, device, OUTPUT_DIR)
