import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import random
import glob
from torchvision import transforms
from PIL import Image
import os
import zipfile

configue = {
    "batch_size": 4,
    "seed": 42,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "start_lr": 2e-4,
    "temp": 10,
    "idt_coef": 0.5,
    "decay_epoch": 0,
    "training": True
}
def extract():
    # 解压缩
    dir_path = r".\data.zip"
    extract_path = r".\data"

    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(dir_path, "r") as f:
        f.extractall(extract_path)

    print("解压缩完成")
    monet_path = os.path.join(extract_path, "data", "monet_jpg")
    photo_path = os.path.join(extract_path, "data", "photo_jpg")
    return monet_path,photo_path

def reverse_img(img,mean=[0.5]*3,std=[0.5]*3):
    for img_channel,mean_channel,std_channel in zip(img,mean,std):
        img_channel.mul_(std_channel).add_(mean_channel)
    return img

# 预测和显示函数
def show_img(transformed_imgs, true_imgs):
    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.imshow(transformed_imgs[-1])
    plt.axis("off")
    plt.title("Generated")

    plt.subplot(122)
    plt.imshow(true_imgs[-1])
    plt.axis("off")
    plt.title("True")

    plt.tight_layout()
    plt.show()

def show_chart(net):
    series=np.arange(len(net.record_metric[0]))
    plt.plot(series,net.record_metric[0],"r-",label="G_loss")
    plt.plot(series,net.record_metric[1],"b-",label="D_loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("loss curve")
    plt.savefig("./Storage/loss curve")
    plt.show()

class PhotoDataset(Dataset):
    def __init__(self, photo_path, size=(256, 256), configue=configue):
        super().__init__()
        random.seed(None)
        self.photo_path = glob.glob(photo_path + "/*")
        random.shuffle(self.photo_path)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        random.seed(configue["seed"])

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, idx):
        photo_img = self.transforms(Image.open(self.photo_path[idx]))
        return photo_img

@torch.no_grad
def Transformer_image(net, data_iter, configue=configue, num=1):
    net.G_Monet2Photo.eval()
    net.G_Photo2Monet.eval()
    net.D_Monet.eval()
    net.D_Photo.eval()
    true_imgs, transformed_imgs = [], []
    for i, img in enumerate(data_iter):
        if i == num:
            break
        device = configue["device"]
        img = img.to(device)
        transformed_img = net.G_Photo2Monet(img)
        transformed_img = reverse_img(transformed_img.cpu()[0])
        transformed_img = transforms.ToPILImage()(transformed_img).convert("RGB")
        transformed_imgs.append(transformed_img)

        img = reverse_img(img.cpu())
        img = transforms.ToPILImage()(img[0]).convert("RGB")
        true_imgs.append(img)

        transformed_img.save(f"Image/image_{i}.png")
    return transformed_imgs, true_imgs

def load(net):
    net.G_Monet2Photo.load_state_dict(torch.load(".\Storage\G_Monet2Photo.pth"))
    net.G_Photo2Monet.load_state_dict(torch.load(".\Storage\G_Photo2Monet.pth"))

    return net.G_Monet2Photo,net.G_Photo2Monet

def save(net):
    torch.save(net.G_Monet2Photo.state_dict(),".\Storage\G_Monet2Photo.pth")
    torch.save(net.G_Photo2Monet.state_dict(), ".\Storage\G_Photo2Monet.pth")


monet_path,photo_path=extract()
photo_iter = DataLoader(PhotoDataset(photo_path), batch_size=1, shuffle=True, pin_memory=True)
