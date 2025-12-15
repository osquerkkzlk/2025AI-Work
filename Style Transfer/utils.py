from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch
import base64

class Recorder:
    def __init__(self, num):
        self.metric = [[] for _ in range(num)]
    def __getitem__(self, item):
        return self.metric[item]
    def add(self, *args):
        for i, arg in enumerate(args):
            self.metric[i].append(arg)

def load_image(img_path,max_size=256):
    image=Image.open(img_path)
    size=min(max(image.size),max_size)
    transform=transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0)

def convert_image(img):
    image=img.to("cpu").clone().detach()
    image=image.squeeze().permute(1,2,0)
    image=image*torch.tensor((0.229,0.224,0.225))+torch.tensor((0.485,0.456,0.406))
    return transforms.ToPILImage()(image.clip(0,1).permute(2,0,1))

def display_loss(contents_l, style_l, tv_l, sum_l):
    plt.figure(figsize=(10, 6))
    plt.plot(contents_l, label="Content Loss", color="r")
    plt.plot(style_l, label="Style Loss", color="b")
    plt.plot(tv_l, label="TV Loss", color="c")
    plt.plot(sum_l, label="Total Loss", color="m")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.savefig("./Storage/loss_curve.png")
    plt.show()

def save(imgs, configue):
    print("Saving images...")
    for i, img in enumerate(imgs):
        temp_img = convert_image(img)
        temp_img.save(f"./Storage/{(i+1)*configue['epoch_step']}_{configue['epochs']}.png")


